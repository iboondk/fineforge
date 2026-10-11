// Some helpers here (canonical_text, normalize, Message fields) are consumed by
// later milestones (dedupe, shard), so they are not all referenced by main yet.
#![allow(dead_code)]

use serde_json::Value;

/// One dataset record in either accepted shape (see PLAN.md section 3).
#[derive(Debug, Clone, PartialEq)]
pub enum Record {
    Chat {
        messages: Vec<Message>,
    },
    Instruction {
        instruction: String,
        input: String,
        output: String,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Message {
    pub role: String,
    pub content: String,
}

const ROLES: [&str; 3] = ["system", "user", "assistant"];

/// Parses one JSONL line. Returns the typed record plus the raw JSON value so
/// that extra top-level keys can be preserved by later commands.
pub fn parse_line(line: &str) -> Result<(Record, Value), String> {
    let value: Value = serde_json::from_str(line).map_err(|e| format!("invalid JSON: {e}"))?;
    let obj = value
        .as_object()
        .ok_or_else(|| "record is not a JSON object".to_string())?;

    let record = if let Some(messages) = obj.get("messages") {
        Record::Chat {
            messages: parse_chat(messages)?,
        }
    } else if let Some(instruction) = obj.get("instruction") {
        parse_instruction(instruction, obj.get("input"), obj.get("output"))?
    } else {
        return Err("unknown record shape".to_string());
    };

    Ok((record, value))
}

fn parse_chat(messages: &Value) -> Result<Vec<Message>, String> {
    let arr = messages
        .as_array()
        .ok_or_else(|| "messages must be an array".to_string())?;
    if arr.len() < 2 {
        return Err("chat needs at least 2 messages".to_string());
    }

    let mut out = Vec::with_capacity(arr.len());
    for (i, item) in arr.iter().enumerate() {
        let role = item
            .get("role")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("message {i} role must be a string"))?;
        if !ROLES.contains(&role) {
            return Err(format!("message {i} has invalid role '{role}'"));
        }
        if role == "system" && i != 0 {
            return Err(format!(
                "system message only allowed at index 0 (found at {i})"
            ));
        }
        let content = item
            .get("content")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("message {i} content must be a string"))?;
        if content.trim().is_empty() {
            return Err(format!("message {i} content is empty"));
        }
        out.push(Message {
            role: role.to_string(),
            content: content.to_string(),
        });
    }

    if !out.iter().any(|m| m.role == "user") {
        return Err("chat needs at least one user message".to_string());
    }
    if out.last().map(|m| m.role.as_str()) != Some("assistant") {
        return Err("last message must be assistant".to_string());
    }
    Ok(out)
}

fn parse_instruction(
    instruction: &Value,
    input: Option<&Value>,
    output: Option<&Value>,
) -> Result<Record, String> {
    let instruction = non_empty_str(instruction, "instruction")?;

    let input = match input {
        None => String::new(),
        Some(v) => v
            .as_str()
            .ok_or_else(|| "input must be a string when present".to_string())?
            .to_string(),
    };

    let output = match output {
        None => return Err("output is required".to_string()),
        Some(v) => non_empty_str(v, "output")?,
    };

    Ok(Record::Instruction {
        instruction,
        input,
        output,
    })
}

fn non_empty_str(v: &Value, field: &str) -> Result<String, String> {
    match v.as_str() {
        Some(s) if !s.trim().is_empty() => Ok(s.to_string()),
        _ => Err(format!("{field} must be a non-empty string")),
    }
}

/// Canonical text used for dedupe hashing. Must stay stable (PLAN.md section 3).
pub fn canonical_text(r: &Record) -> String {
    match r {
        Record::Chat { messages } => messages
            .iter()
            .map(|m| format!("{}: {}", m.role, m.content))
            .collect::<Vec<_>>()
            .join("\n"),
        Record::Instruction {
            instruction,
            input,
            output,
        } => format!("instruction: {instruction}\ninput: {input}\noutput: {output}"),
    }
}

/// Lowercases, collapses every run of Unicode whitespace to one space, and trims.
pub fn normalize(s: &str) -> String {
    s.to_lowercase()
        .split(char::is_whitespace)
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok(line: &str) -> Record {
        parse_line(line).expect("expected valid record").0
    }

    fn err(line: &str) -> String {
        parse_line(line).expect_err("expected invalid record")
    }

    const GOOD_CHAT: &str =
        r#"{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"yo"}]}"#;

    #[test]
    fn rejects_non_json() {
        assert!(err("not json").starts_with("invalid JSON"));
    }

    #[test]
    fn rejects_non_object() {
        assert_eq!(err("[1,2]"), "record is not a JSON object");
    }

    #[test]
    fn rejects_unknown_shape() {
        assert_eq!(err(r#"{"text":"x"}"#), "unknown record shape");
    }

    #[test]
    fn accepts_minimal_chat() {
        match ok(GOOD_CHAT) {
            Record::Chat { messages } => {
                assert_eq!(messages.len(), 2);
                assert_eq!(messages[0].role, "user");
                assert_eq!(messages[1].content, "yo");
            }
            other => panic!("expected chat, got {other:?}"),
        }
    }

    #[test]
    fn chat_needs_at_least_two_messages() {
        let line = r#"{"messages":[{"role":"user","content":"hi"}]}"#;
        assert_eq!(err(line), "chat needs at least 2 messages");
    }

    #[test]
    fn chat_rejects_unknown_role() {
        let line =
            r#"{"messages":[{"role":"tool","content":"x"},{"role":"assistant","content":"y"}]}"#;
        assert_eq!(err(line), "message 0 has invalid role 'tool'");
    }

    #[test]
    fn chat_role_must_be_string() {
        let line = r#"{"messages":[{"role":1,"content":"x"},{"role":"assistant","content":"y"}]}"#;
        assert_eq!(err(line), "message 0 role must be a string");
    }

    #[test]
    fn chat_content_must_be_non_empty_string() {
        let blank =
            r#"{"messages":[{"role":"user","content":"   "},{"role":"assistant","content":"y"}]}"#;
        assert_eq!(err(blank), "message 0 content is empty");

        let not_str =
            r#"{"messages":[{"role":"user","content":5},{"role":"assistant","content":"y"}]}"#;
        assert_eq!(err(not_str), "message 0 content must be a string");
    }

    #[test]
    fn chat_needs_a_user_message() {
        let line =
            r#"{"messages":[{"role":"system","content":"s"},{"role":"assistant","content":"y"}]}"#;
        assert_eq!(err(line), "chat needs at least one user message");
    }

    #[test]
    fn chat_last_message_must_be_assistant() {
        let line = r#"{"messages":[{"role":"user","content":"a"},{"role":"assistant","content":"b"},{"role":"user","content":"c"}]}"#;
        assert_eq!(err(line), "last message must be assistant");
    }

    #[test]
    fn system_only_allowed_at_index_zero() {
        let bad = r#"{"messages":[{"role":"user","content":"a"},{"role":"system","content":"s"},{"role":"user","content":"b"},{"role":"assistant","content":"c"}]}"#;
        assert_eq!(
            err(bad),
            "system message only allowed at index 0 (found at 1)"
        );

        let good = r#"{"messages":[{"role":"system","content":"s"},{"role":"user","content":"a"},{"role":"assistant","content":"c"}]}"#;
        assert!(parse_line(good).is_ok());
    }

    #[test]
    fn chat_messages_must_be_array() {
        assert_eq!(err(r#"{"messages":"hi"}"#), "messages must be an array");
    }

    #[test]
    fn accepts_instruction_with_input() {
        let line = r#"{"instruction":"Translate","input":"Hi","output":"Salut"}"#;
        assert_eq!(
            ok(line),
            Record::Instruction {
                instruction: "Translate".into(),
                input: "Hi".into(),
                output: "Salut".into(),
            }
        );
    }

    #[test]
    fn instruction_input_defaults_to_empty() {
        let line = r#"{"instruction":"Name a color","output":"Red"}"#;
        assert_eq!(
            ok(line),
            Record::Instruction {
                instruction: "Name a color".into(),
                input: String::new(),
                output: "Red".into(),
            }
        );
    }

    #[test]
    fn instruction_input_must_be_string_when_present() {
        let line = r#"{"instruction":"x","input":3,"output":"y"}"#;
        assert_eq!(err(line), "input must be a string when present");
    }

    #[test]
    fn instruction_needs_non_empty_instruction() {
        let blank = r#"{"instruction":" ","output":"y"}"#;
        assert_eq!(err(blank), "instruction must be a non-empty string");

        let not_str = r#"{"instruction":true,"output":"y"}"#;
        assert_eq!(err(not_str), "instruction must be a non-empty string");
    }

    #[test]
    fn instruction_needs_non_empty_output() {
        let empty = r#"{"instruction":"x","input":"","output":"  "}"#;
        assert_eq!(err(empty), "output must be a non-empty string");

        let missing = r#"{"instruction":"x"}"#;
        assert_eq!(err(missing), "output is required");
    }

    #[test]
    fn parse_line_returns_raw_value_with_extra_keys() {
        let line = r#"{"instruction":"x","output":"y","id":"abc"}"#;
        let (_, raw) = parse_line(line).unwrap();
        assert_eq!(raw["id"], "abc");
    }

    #[test]
    fn canonical_text_for_chat() {
        let r = ok(GOOD_CHAT);
        assert_eq!(canonical_text(&r), "user: hi\nassistant: yo");
    }

    #[test]
    fn canonical_text_for_instruction() {
        let r = Record::Instruction {
            instruction: "Translate".into(),
            input: "Hi".into(),
            output: "Salut".into(),
        };
        assert_eq!(
            canonical_text(&r),
            "instruction: Translate\ninput: Hi\noutput: Salut"
        );
    }

    #[test]
    fn canonical_text_uses_empty_input_when_missing() {
        let (r, _) = parse_line(r#"{"instruction":"x","output":"y"}"#).unwrap();
        assert_eq!(canonical_text(&r), "instruction: x\ninput: \noutput: y");
    }

    #[test]
    fn normalize_lowercases_collapses_and_trims() {
        assert_eq!(normalize("  Hello\t\nWORLD  "), "hello world");
    }

    #[test]
    fn normalize_handles_unicode_whitespace() {
        assert_eq!(normalize("a\u{00A0}\u{2003}b"), "a b");
    }

    #[test]
    fn normalize_empty_and_whitespace_only() {
        assert_eq!(normalize(""), "");
        assert_eq!(normalize(" \t\n "), "");
    }
}
