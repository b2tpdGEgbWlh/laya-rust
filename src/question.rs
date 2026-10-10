//! Question rendering and sequence layout.
//!
//! The layout below must stay byte-for-byte with the reference `build_sequence` in
//! `rl_common.py`: the encoder is trained against that exact geometry, so any drift changes
//! what the head reads at the marker positions.
//!
//! ```text
//! [CLS] <type> question: <instructions> [SEP]
//!   [MASK] opt1 [MASK] opt2 ... [MASK] optK
//! [SEP]
//! <serialized state>
//! [SEP]
//! ```

use crate::question::QType::*;
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokenizers::Tokenizer;

/// Options share `head_max_len`; the per-option shrink floor keeps a label readable after
/// many-option questions are squeezed. Matches the reference's hardcoded cap and floor.
const MAX_OPTION_TOKENS: usize = 48;

#[derive(Debug, Clone)]
pub(crate) struct SpecialIds {
    pub cls: u32,
    pub sep: u32,
    pub mask: u32,
    pub pad: u32,
    pub mask_text: String,
}

/// A typed question.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub struct Question {
    /// A stable answer key in the response and a `qid` for question-type embeddings.
    pub id: String,
    /// One of `choice`, `score`, `noul`.
    #[serde(rename = "type")]
    pub qtype: QType,
    /// The full instruction to the model.
    pub instructions: String,
    /// `choice`: label → description; `score`/`noul`: optional label → description.
    #[serde(default)]
    pub criteria: Option<serde_json::Map<String, Value>>,
    /// `score`: ordered level descriptions; `noul`: two-sided criteria.
    #[serde(default = "Option::default")]
    pub levels: Option<Vec<String>>,
}

impl Question {
    /// Raw IDs are for humans; the tokenized label order is the reported distribution order.
    pub fn choice(id: &str, labels: impl IntoIterator<Item = impl Into<String>>) -> Self {
        let labels: Vec<String> = labels.into_iter().map(Into::into).collect();
        let criteria = labels
            .into_iter()
            .map(|l| (l, Value::Null))
            .collect::<serde_json::Map<_, _>>();
        Question {
            id: id.to_string(),
            qtype: QType::Choice,
            instructions: String::new(),
            criteria: Some(criteria),
            levels: None,
        }
    }

    /// `levels` are ordered low → high.
    pub fn score(id: &str, levels: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Question {
            id: id.to_string(),
            qtype: QType::Score,
            instructions: String::new(),
            criteria: None,
            levels: Some(levels.into_iter().map(Into::into).collect()),
        }
    }

    /// Irony-optional. If `criteria` is empty, the default sides are `false` then `true`.
    pub fn noul(id: &str, statement: impl Into<String>) -> Self {
        Question {
            id: id.to_string(),
            instructions: statement.into(),
            qtype: QType::Noul,
            criteria: None,
            levels: None,
        }
    }

    /// The rendered option texts, in label order, exactly as the encoder sees them.
    /// Mirror of `render_options` in `rl_common.py`.
    pub fn render_options(&self) -> Result<Vec<String>> {
        let opts = match self.qtype {
            Choice => {
                let map = self.criteria.clone().ok_or_else(|| {
                    anyhow::anyhow!("choice question {id:?} has no `criteria`", id = self.id)
                })?;
                map.into_iter()
                    .map(|(k, v)| {
                        if v.is_null() || v.as_str().map(str::is_empty).unwrap_or(false) {
                            k
                        } else {
                            format!("{k}: {v}")
                        }
                    })
                    .collect()
            }
            Score => {
                let levels = self.levels.clone().ok_or_else(|| {
                    anyhow::anyhow!("score question {id:?} has no `levels`", id = self.id)
                })?;
                levels
                    .into_iter()
                    .enumerate()
                    .map(|(i, l)| {
                        if l.trim().is_empty() {
                            format!("level {i}")
                        } else {
                            format!("level {i}: {l}")
                        }
                    })
                    .collect()
            }
            Noul => {
                let c = self.criteria.clone().unwrap_or_default();
                if c.is_empty() {
                    vec!["false".to_string(), "true".to_string()]
                } else {
                    let f = c.get("false").map(String::as_str).unwrap_or("");
                    let t = c.get("true").map(String::as_str).unwrap_or("");
                    vec![format!("false: {f}"), format!("true: {t}")]
                }
            }
        };
        Ok(opts)
    }

    /// The answer keys for a `choice` question, in label order.
    pub fn choice_keys(&self) -> Result<Vec<String>> {
        let map = self.criteria.clone().ok_or_else(|| {
            anyhow::anyhow!("choice question {id:?} has no `criteria`", id = self.id)
        })?;
        Ok(map.into_keys().collect())
    }

    /// The `instructions` or, for `noul`, the statement.
    pub fn instructions_text(&self) -> String {
        if self.instructions.is_empty() {
            self.id.clone()
        } else {
            self.instructions.clone()
        }
    }

    /// The rendered legend for a `score` answer, in level order.
    pub fn score_levels(&self) -> Vec<String> {
        self.levels.clone().unwrap_or_default()
    }
}

/// The three question types. Indices line up with the checkpoint's embedding rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QType {
    Choice = 0,
    Score = 1,
    Noul = 2,
}

impl QType {
    pub fn index(&self) -> usize {
        *self as usize
    }
    pub fn as_str(&self) -> &'static str {
        match self {
            Choice => "choice",
            Score => "score",
            Noul => "noul",
        }
    }
    pub fn from_index(i: usize) -> Self {
        match i {
            0 => Choice,
            1 => Score,
            _ => Noul,
        }
    }
}

/// The encoded sequence plus everything the model needs to read it back.
#[derive(Debug, Clone)]
pub(crate) struct Encoded {
    pub ids: Vec<u32>,
    pub markers: Vec<usize>,
    /// Index of the `[SEP]` that closes the instructions.
    pub head_sep: usize,
    /// Index of the `[SEP]` that closes the option block.
    pub opts_sep: usize,
    /// Number of option labels whose text was cut to fit `head_max_len`.
    pub labels_clipped: usize,
}

// The encoder is byte-sensitive (it scores individual markers), so the layout is fixed here:
// no XML, no peer instructions. Keep the Cargo.toml description in sync with this.

fn encode_plain(tok: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    tok.encode(text, false)
        .map(|e| e.get_ids().to_vec())
        .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))
}

/// Shrink every option evenly to fit the head budget, matching the reference. Returns the number
/// of labels whose text was actually cut (`per` is floored at 4, so short labels survive intact).
fn shrink_options(opt_ids: &mut [Vec<u32>], head_max_len: usize) -> usize {
    let used: usize = opt_ids.iter().map(|o| o.len()).sum();
    if head_max_len as isize - used as isize >= 16 {
        return 0;
    }
    // Too many or too long options: shrink every option evenly so the instructions still fit.
    let per = std::cmp::max(
        4,
        head_max_len.saturating_sub(16) / std::cmp::max(1, opt_ids.len()),
    );
    let mut clipped = 0;
    for o in opt_ids.iter_mut() {
        if o.len() > per {
            o.truncate(per);
            clipped += 1;
        }
    }
    clipped
}

/// Cut a tokenized state to `room` tokens. `keep_end` keeps the tail (latest content); the
/// default keeps the head. Matches `st[:room]` / `st[-room:]` in the reference.
fn truncate_state(st: &mut Vec<u32>, room: usize, keep_end: bool) {
    if keep_end {
        if st.len() > room {
            *st = st.split_off(st.len() - room);
        }
    } else {
        st.truncate(room);
    }
}

/// Build the input sequence and the marker positions for one question.
pub(crate) fn build_sequence(
    tok: &Tokenizer,
    sp: &SpecialIds,
    state: &Value,
    q: &Question,
    max_len: usize,
    head_max_len: usize,
    truncate_left: bool,
) -> Result<Encoded> {
    let opts = q.render_options()?;
    if opts.is_empty() {
        bail!(
            "a {} question needs at least one option in `criteria`",
            q.qtype.as_str()
        );
    }
    // A literal [MASK] in user text would be read as a marker, so neutralize it, as the reference does.
    let scrub = |s: &str| s.replace(&sp.mask_text, " ");

    let ins = scrub(&q.instructions_text());
    let mut head_ids = encode_plain(tok, &format!("{} question: {}", q.qtype.as_str(), ins))?;

    let mut opt_ids: Vec<Vec<u32>> = Vec::with_capacity(opts.len());
    for o in &opts {
        let mut ids = vec![sp.mask];
        let mut body = encode_plain(tok, &format!(" {}", scrub(o)))?;
        body.truncate(MAX_OPTION_TOKENS);
        ids.extend(body);
        opt_ids.push(ids);
    }

    let labels_clipped = shrink_options(&mut opt_ids, head_max_len);
    let used: usize = opt_ids.iter().map(|o| o.len()).sum();
    let opt_budget = head_max_len as isize - used as isize;
    head_ids.truncate(std::cmp::max(8, opt_budget.max(0) as usize));

    let mut ids = Vec::with_capacity(max_len);
    ids.push(sp.cls);
    ids.extend_from_slice(&head_ids);
    let head_sep = ids.len();
    ids.push(sp.sep);

    let mut markers = Vec::with_capacity(opt_ids.len());
    for o in &opt_ids {
        markers.push(ids.len());
        ids.extend_from_slice(o);
    }
    let opts_sep = ids.len();
    ids.push(sp.sep);

    let room = max_len.saturating_sub(ids.len() + 1);
    let mut st = encode_plain(tok, &scrub(&serialize_state(state)))?;
    truncate_state(&mut st, room, truncate_left);
    ids.extend_from_slice(&st);
    ids.push(sp.sep);
    ids.truncate(max_len);

    let markers: Vec<usize> = markers.into_iter().filter(|m| *m < max_len).collect();
    if markers.len() != opts.len() {
        bail!(
            "options do not fit in head_max_len={head_max_len} tokens ({} of {} markers kept)",
            markers.len(),
            opts.len()
        );
    }
    Ok(Encoded {
        ids,
        markers,
        head_sep,
        opts_sep,
        labels_clipped,
    })
}

// The state is dense prose; the encoder has never seen our internal docstrings.
// Serialize JSON with a compact, deterministic ordering so layout stays stable.
// Encode the state as the model expects: a JSON document when it is one, else the raw text.
fn serialize_state(state: &Value) -> String {
    match state {
        Value::String(s) => s.clone(),
        _ => serde_json::to_string(state).unwrap_or_default(),
    }
}

#[cfg(test)]
mod tests {
    use super::{shrink_options, truncate_state};

    #[test]
    fn shrink_options_is_a_no_op_when_the_head_budget_has_room() {
        let mut opts = vec![vec![0, 1, 2, 3], vec![0, 4, 5]];
        assert_eq!(shrink_options(&mut opts, 64), 0);
        assert_eq!(opts[0].len(), 4);
        assert_eq!(opts[1].len(), 3);
    }

    #[test]
    fn shrink_options_clips_deep_labels_to_the_floor_and_counts_them() {
        let mut opts: Vec<Vec<u32>> = vec![
            vec![0, 10, 11, 12, 13, 14, 15, 16, 17, 18],
            vec![0, 20, 21, 22, 23, 24, 25, 26],
            vec![0, 30],
        ];
        let clipped = shrink_options(&mut opts, 16);
        assert_eq!(clipped, 2);
        assert_eq!(opts[0].len(), 4);
        assert_eq!(opts[1].len(), 4);
        // A label that already fits the floor is left alone.
        assert_eq!(opts[2].len(), 2);
    }

    #[test]
    fn truncate_state_keeps_the_head_by_default() {
        let mut st = vec![1, 2, 3, 4, 5];
        truncate_state(&mut st, 3, false);
        assert_eq!(st, vec![1, 2, 3]);
    }

    #[test]
    fn truncate_state_keeps_the_tail_with_keep_end() {
        let mut st = vec![1, 2, 3, 4, 5];
        truncate_state(&mut st, 3, true);
        assert_eq!(st, vec![3, 4, 5]);
    }

    #[test]
    fn truncate_state_is_a_no_op_when_it_fits() {
        let mut st = vec![1, 2, 3];
        truncate_state(&mut st, 10, true);
        assert_eq!(st, vec![1, 2, 3]);
    }
}
