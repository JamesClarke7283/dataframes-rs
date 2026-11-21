//! pandas-style expression strings: `df.query("a > 5 and b < 3.0")`.
//!
//! The grammar follows `DataFrame.query` / `DataFrame.eval`:
//!
//! ```text
//! or      := and (("or" | "|" | "||") and)*
//! and     := not (("and" | "&" | "&&") not)*
//! not     := ("not" | "~" | "!") not | compare
//! compare := sum (("==" | "!=" | "<" | "<=" | ">" | ">=") sum)?
//!          | sum ["not"] "in" "[" literal ("," literal)* "]"
//! sum     := term (("+" | "-") term)*
//! term    := unary (("*" | "/" | "//" | "%") unary)*
//! unary   := "-" unary | atom
//! atom    := number | string | true | false | null | name | `quoted name` | "(" or ")"
//! ```
//!
//! As in pandas, `&` and `|` have the precedence of `and` and `or`, and `/`
//! is true division.

use super::dsl::{Expr, LiteralValue, col, lit};
use crate::datatypes::DataType;
use crate::error::{PolarsError, PolarsResult};
use crate::frame::DataFrame;
use crate::series::Series;

#[derive(Clone, Debug, PartialEq)]
enum Token {
    Ident(String),
    Int(i128),
    Float(f64),
    Str(String),
    Op(&'static str),
    End,
}

fn syntax(msg: impl std::fmt::Display, source: &str) -> PolarsError {
    PolarsError::ComputeError(format!("invalid query {source:?}: {msg}").into())
}

fn tokenize(source: &str) -> PolarsResult<Vec<Token>> {
    const OPS: [&str; 22] =
        ["==", "!=", "<=", ">=", "&&", "||", "//", "<", ">", "+", "-", "*", "/", "%", "(", ")", "[", "]", ",", "&", "|", "~"];
    let chars: Vec<char> = source.chars().collect();
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
        } else if c.is_ascii_digit() || (c == '.' && chars.get(i + 1).is_some_and(|d| d.is_ascii_digit())) {
            let start = i;
            let mut is_float = false;
            while i < chars.len() {
                match chars[i] {
                    d if d.is_ascii_digit() || d == '_' => i += 1,
                    '.' if !is_float => {
                        is_float = true;
                        i += 1;
                    }
                    'e' | 'E' if chars.get(i + 1).is_some_and(|d| d.is_ascii_digit() || *d == '-' || *d == '+') => {
                        is_float = true;
                        i += 2;
                    }
                    _ => break,
                }
            }
            let text: String = chars[start..i].iter().filter(|c| **c != '_').collect();
            tokens.push(if is_float {
                Token::Float(text.parse().map_err(|_| syntax(format!("bad number {text:?}"), source))?)
            } else {
                Token::Int(text.parse().map_err(|_| syntax(format!("bad number {text:?}"), source))?)
            });
        } else if c == '"' || c == '\'' || c == '`' {
            let mut text = String::new();
            i += 1;
            loop {
                match chars.get(i) {
                    None => return Err(syntax("unterminated quote", source)),
                    Some(&q) if q == c => break,
                    Some('\\') if c != '`' && i + 1 < chars.len() => {
                        text.push(chars[i + 1]);
                        i += 2;
                    }
                    Some(&other) => {
                        text.push(other);
                        i += 1;
                    }
                }
            }
            i += 1;
            tokens.push(if c == '`' { Token::Ident(text) } else { Token::Str(text) });
        } else if c.is_alphabetic() || c == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            tokens.push(Token::Ident(chars[start..i].iter().collect()));
        } else if c == '!' && chars.get(i + 1) != Some(&'=') {
            tokens.push(Token::Op("~"));
            i += 1;
        } else {
            let rest: String = chars[i..chars.len().min(i + 2)].iter().collect();
            match OPS.iter().find(|op| rest.starts_with(**op)) {
                Some(op) => {
                    tokens.push(Token::Op(op));
                    i += op.chars().count();
                }
                None => return Err(syntax(format!("unexpected character {c:?}"), source)),
            }
        }
    }
    tokens.push(Token::End);
    Ok(tokens)
}

struct Parser<'a> {
    source: &'a str,
    tokens: Vec<Token>,
    pos: usize,
    /// Backtick-quoted identifiers are always columns, even if they spell a keyword.
    quoted: Vec<bool>,
}

impl Parser<'_> {
    fn peek(&self) -> &Token {
        &self.tokens[self.pos]
    }

    fn next(&mut self) -> Token {
        let t = self.tokens[self.pos].clone();
        if self.pos + 1 < self.tokens.len() {
            self.pos += 1;
        }
        t
    }

    fn eat_op(&mut self, ops: &[&str]) -> Option<&'static str> {
        if let Token::Op(op) = *self.peek() {
            if ops.contains(&op) {
                self.pos += 1;
                return Some(op);
            }
        }
        None
    }

    fn is_keyword(&self, word: &str) -> bool {
        matches!(self.peek(), Token::Ident(w) if w == word) && !self.quoted[self.pos]
    }

    fn eat_keyword(&mut self, word: &str) -> bool {
        let hit = self.is_keyword(word);
        if hit {
            self.pos += 1;
        }
        hit
    }

    fn err<T>(&self, msg: impl std::fmt::Display) -> PolarsResult<T> {
        Err(syntax(msg, self.source))
    }

    fn or(&mut self) -> PolarsResult<Expr> {
        let mut left = self.and()?;
        while self.eat_keyword("or") || self.eat_op(&["|", "||"]).is_some() {
            left = left.or(self.and()?);
        }
        Ok(left)
    }

    fn and(&mut self) -> PolarsResult<Expr> {
        let mut left = self.not()?;
        while self.eat_keyword("and") || self.eat_op(&["&", "&&"]).is_some() {
            left = left.and(self.not()?);
        }
        Ok(left)
    }

    fn not(&mut self) -> PolarsResult<Expr> {
        if self.eat_keyword("not") || self.eat_op(&["~"]).is_some() {
            return Ok(self.not()?.not());
        }
        self.compare()
    }

    fn compare(&mut self) -> PolarsResult<Expr> {
        let left = self.sum()?;
        if let Some(op) = self.eat_op(&["==", "!=", "<=", ">=", "<", ">"]) {
            let right = self.sum()?;
            return Ok(match op {
                "==" => left.eq(right),
                "!=" => left.neq(right),
                "<" => left.lt(right),
                "<=" => left.lt_eq(right),
                ">" => left.gt(right),
                _ => left.gt_eq(right),
            });
        }
        // `x in [..]` / `x not in [..]`
        let negated = self.is_keyword("not") && matches!(self.tokens.get(self.pos + 1), Some(Token::Ident(w)) if w == "in");
        if negated {
            self.pos += 1;
        }
        if self.eat_keyword("in") {
            let close = match self.eat_op(&["[", "("]) {
                Some("[") => "]",
                Some(_) => ")",
                None => return self.err("expected a list after `in`"),
            };
            let mut values = Vec::new();
            while self.eat_op(&[close]).is_none() {
                values.push(self.sum()?);
                if self.eat_op(&[","]).is_none() && !matches!(self.peek(), Token::Op(op) if *op == close) {
                    return self.err("expected `,` or the end of the list");
                }
            }
            let contained = left.is_in(values);
            return Ok(if negated { contained.not() } else { contained });
        }
        Ok(left)
    }

    fn sum(&mut self) -> PolarsResult<Expr> {
        let mut left = self.term()?;
        while let Some(op) = self.eat_op(&["+", "-"]) {
            let right = self.term()?;
            left = if op == "+" { left + right } else { left - right };
        }
        Ok(left)
    }

    fn term(&mut self) -> PolarsResult<Expr> {
        let mut left = self.unary()?;
        while let Some(op) = self.eat_op(&["*", "//", "/", "%"]) {
            let right = self.unary()?;
            left = match op {
                "*" => left * right,
                // pandas' `/` is true division.
                "/" => left.cast(DataType::Float64) / right,
                "//" => left / right,
                _ => left % right,
            };
        }
        Ok(left)
    }

    fn unary(&mut self) -> PolarsResult<Expr> {
        if self.eat_op(&["-"]).is_some() {
            return Ok(match self.unary()? {
                Expr::Literal(LiteralValue::Int(i)) => lit_int(-i),
                Expr::Literal(LiteralValue::Float(f)) => lit(-f),
                other => -other,
            });
        }
        self.atom()
    }

    fn atom(&mut self) -> PolarsResult<Expr> {
        let quoted = self.quoted[self.pos];
        match self.next() {
            Token::Int(i) => Ok(lit_int(i)),
            Token::Float(f) => Ok(lit(f)),
            Token::Str(s) => Ok(lit(s)),
            Token::Ident(name) if !quoted => Ok(match name.as_str() {
                "true" | "True" => lit(true),
                "false" | "False" => lit(false),
                "null" | "None" | "NaN" if name != "NaN" => Expr::Literal(LiteralValue::Null),
                "NaN" | "nan" => lit(f64::NAN),
                "and" | "or" | "not" | "in" => return self.err(format!("unexpected keyword `{name}`")),
                _ => col(name.as_str()),
            }),
            Token::Ident(name) => Ok(col(name.as_str())),
            Token::Op("(") => {
                let inner = self.or()?;
                if self.eat_op(&[")"]).is_none() {
                    return self.err("expected `)`");
                }
                Ok(inner)
            }
            Token::Op(op) => self.err(format!("unexpected `{op}`")),
            Token::End => self.err("unexpected end of input"),
        }
    }
}

fn lit_int(i: i128) -> Expr {
    Expr::Literal(LiteralValue::Int(i))
}

/// Parse a pandas-style expression string into an [`Expr`].
///
/// ```
/// use dataframes::prelude::*;
/// let e = parse_query("a > 5 and not (name == 'x')")?;
/// assert_eq!(e, col("a").gt(5).and(col("name").eq("x").not()));
/// # Ok::<(), PolarsError>(())
/// ```
pub fn parse_query(source: &str) -> PolarsResult<Expr> {
    let tokens = tokenize(source)?;
    // Recover which identifiers were backtick-quoted: re-scan the source in step.
    let mut quoted = Vec::with_capacity(tokens.len());
    let mut rest = source;
    for t in &tokens {
        rest = rest.trim_start();
        quoted.push(matches!(t, Token::Ident(_)) && rest.starts_with('`'));
        let width = match t {
            Token::Ident(name) if rest.starts_with('`') => name.len() + 2,
            Token::Ident(name) => name.len(),
            Token::Str(_) => {
                let quote = rest.chars().next().unwrap_or('"');
                let mut end = 1;
                let bytes = rest.as_bytes();
                while end < bytes.len() && bytes[end] as char != quote {
                    end += if bytes[end] == b'\\' { 2 } else { 1 };
                }
                (end + 1).min(rest.len())
            }
            Token::Op(op) if *op == "~" && rest.starts_with('!') => 1,
            Token::Op(op) => op.len(),
            Token::Int(_) | Token::Float(_) => rest
                .char_indices()
                .scan(false, |seen_e, (i, c)| {
                    let ok = c.is_ascii_digit() || c == '_' || c == '.' || c == 'e' || c == 'E' || (*seen_e && (c == '-' || c == '+'));
                    *seen_e = c == 'e' || c == 'E';
                    ok.then_some(i + c.len_utf8())
                })
                .last()
                .unwrap_or(0),
            Token::End => 0,
        };
        rest = rest.get(width..).unwrap_or("");
    }
    let mut parser = Parser { source, tokens, pos: 0, quoted };
    let expr = parser.or()?;
    if *parser.peek() != Token::End {
        return parser.err(format!("unexpected trailing input at token {}", parser.pos + 1));
    }
    Ok(expr)
}

impl DataFrame {
    /// Keep the rows for which a pandas-style expression string holds, as
    /// `df.query("a > 5 and b < 3.0")`.
    ///
    /// ```
    /// use dataframes::prelude::*;
    /// let df = df!("a" => [1, 6, 9], "b" => [1.0, 2.0, 5.0])?;
    /// let out = df.query("a > 5 and b < 3.0")?;
    /// assert_eq!(out.column("a")?.i32()?.cont_slice()?, &[6]);
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn query(&self, expr: &str) -> PolarsResult<DataFrame> {
        self.filter(parse_query(expr)?)
    }

    /// Evaluate a pandas-style expression string to a column, as `df.eval("a + b")`.
    pub fn eval(&self, expr: &str) -> PolarsResult<Series> {
        self.eval_expr(&parse_query(expr)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::df;

    #[test]
    fn parses_precedence_like_pandas() {
        assert_eq!(parse_query("a > 5").unwrap(), col("a").gt(5));
        assert_eq!(parse_query("a>5 and b<3.5").unwrap(), col("a").gt(5).and(col("b").lt(3.5)));
        // `&` binds like `and`, so no parentheses are needed around comparisons.
        assert_eq!(parse_query("a > 5 & b < 3").unwrap(), col("a").gt(5).and(col("b").lt(3)));
        assert_eq!(
            parse_query("a == 1 or b == 2 and c == 3").unwrap(),
            col("a").eq(1).or(col("b").eq(2).and(col("c").eq(3)))
        );
        assert_eq!(parse_query("not a == 1").unwrap(), col("a").eq(1).not());
        assert_eq!(parse_query("~(a == 1) | !(b != 2)").unwrap(), col("a").eq(1).not().or(col("b").neq(2).not()));
        assert_eq!(parse_query("a + b * 2 >= -3").unwrap(), (col("a") + col("b") * 2).gt_eq(-3));
        assert_eq!(parse_query("(a + b) * 2 % 3 <= 1e3").unwrap(), ((col("a") + col("b")) * 2 % 3).lt_eq(1000.0));
        assert_eq!(parse_query("name == 'it\\'s' and `my col` != \"x\"").unwrap(), col("name").eq("it's").and(col("my col").neq("x")));
        assert_eq!(parse_query("a in [1, 2, 3]").unwrap(), col("a").is_in([1, 2, 3]));
        assert_eq!(parse_query("s not in ('x', 'y')").unwrap(), col("s").is_in(["x", "y"]).not());
        assert_eq!(parse_query("flag == True and x != None").unwrap(), col("flag").eq(true).and(col("x").neq(Expr::Literal(LiteralValue::Null))));
        assert_eq!(parse_query("a / b").unwrap(), col("a").cast(DataType::Float64) / col("b"));
        assert_eq!(parse_query("a // 2").unwrap(), col("a") / 2);
        assert_eq!(parse_query("`and` > 1_000").unwrap(), col("and").gt(1000));
        for bad in ["a >", "a > > 1", "(a > 1", "a ? 1", "a in 3", "and", "a > 1 b", "'open"] {
            assert!(parse_query(bad).is_err(), "{bad:?} should not parse");
        }
    }

    #[test]
    fn query_and_eval_on_frames() {
        let df = df!(
            "a" => [1i64, 6, 9, 12],
            "b" => [1.0f64, 2.0, 5.0, 0.5],
            "s" => ["x", "y", "x", "z"],
        )
        .unwrap();
        assert_eq!(df.query("a > 5 and b < 3.0").unwrap()["a"].i64().unwrap().cont_slice().unwrap(), &[6, 12]);
        assert_eq!(df.query("s == 'x' or a == 12").unwrap().height(), 3);
        assert_eq!(df.query("s in ['y', 'z']").unwrap().height(), 2);
        assert_eq!(df.query("not (a % 2 == 0)").unwrap()["a"].i64().unwrap().cont_slice().unwrap(), &[1, 9]);
        assert_eq!(df.query("a * b > 10").unwrap().height(), 2);
        assert_eq!(df.eval("a / 4").unwrap().f64().unwrap().cont_slice().unwrap(), &[0.25, 1.5, 2.25, 3.0]);
        assert_eq!(df.eval("a // 4").unwrap().i64().unwrap().cont_slice().unwrap(), &[0, 1, 2, 3]);
        assert_eq!(df.eval("a + b").unwrap().name(), "a");
        assert!(df.query("missing > 1").is_err());
        assert!(df.query("a +").is_err());
    }
}
