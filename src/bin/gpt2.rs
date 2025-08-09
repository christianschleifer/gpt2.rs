use regex::Regex;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};

use std::{fs, iter};

fn main() {
    let tokenizer = Tokenizer::new();

    let input_string = "Hello world!";

    println!("Tokenizing the input string {input_string}");
    let tokens = tokenizer.tokenize(input_string);
    println!("Got tokens: {tokens:?}");

    println!("Detokenizing the tokens {tokens:?}");
    let output = tokenizer.detokenize(&tokens);
    println!("Got ouput: {output}");
}

type Token = u32;

#[derive(Deserialize, Debug)]
struct ModelSpec {
    vocab: HashMap<String, Token>,
    merges: Vec<String>,
}

#[derive(Deserialize, Debug)]
struct TokenizerSpec {
    model: ModelSpec,
}

#[derive(PartialEq, Eq, Hash, Clone, Debug)]
struct Merge(String, String);

struct Tokenizer {
    mappings: HashMap<String, Token>,
    reverse_mappings: HashMap<Token, String>,
    splitting_regex: Regex,
    byte_mappings: HashMap<u8, char>,
    reverse_byte_mappings: HashMap<char, u8>,
    merges: HashMap<Merge, u32>,
}

impl Tokenizer {
    fn new() -> Self {
        let json = fs::read_to_string("resources/tokenizer.json")
            .expect("could not open file resources/tokenizer.json");

        let tokenizer: TokenizerSpec =
            serde_json::from_str(&json).expect("could not parse file resources/tokenizer.json");

        let mappings = tokenizer.model.vocab.clone();

        let mut reverse_mappings = HashMap::with_capacity(mappings.len());

        for (key, value) in &mappings {
            reverse_mappings.insert(*value, key.clone());
        }

        let merges: HashMap<Merge, u32> = tokenizer
            .model
            .merges
            .into_iter()
            .enumerate()
            .map(|(idx, merge)| {
                let (left, right) = merge.split_once(" ").expect("merges formatted incorrectly");
                (Merge(left.to_owned(), right.to_owned()), idx as u32)
            })
            .collect();

        let splitting_regex =
            Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+")
                .expect("could not compile regexp");

        let byte_mappings = Tokenizer::get_byte_mappings();

        let mut reverse_byte_mappings = HashMap::with_capacity(byte_mappings.len());

        for (key, value) in &byte_mappings {
            reverse_byte_mappings.insert(*value, *key);
        }

        Tokenizer {
            mappings,
            reverse_mappings,
            splitting_regex,
            byte_mappings,
            reverse_byte_mappings,
            merges,
        }
    }

    fn get_byte_mappings() -> HashMap<u8, char> {
        let mut bytes: Vec<u8> = Vec::new();

        for i in b'!'..=b'~' {
            bytes.push(i);
        }

        for i in 0xA1..=0xAC {
            bytes.push(i);
        }

        for i in 0xAE..=0xFF {
            bytes.push(i);
        }

        let mut chars: Vec<char> = bytes.iter().map(|b| *b as char).collect();

        let mut n = 0_u16;
        for i in 0..=255 {
            if !bytes.contains(&i) {
                bytes.push(i);
                let new_char = 2_u16.pow(8) + n;
                let new_char = char::from_u32(new_char as u32).expect("could not convert char");

                chars.push(new_char);
                n += 1;
            }
        }

        iter::zip(bytes, chars).collect()
    }

    fn tokenize(&self, input: &str) -> Vec<Token> {
        let mut tokens: Vec<Token> = Vec::new();
        for part in self.splitting_regex.find_iter(input) {
            let mut chars = Vec::new();
            for byte in part.as_str().bytes() {
                chars.push(
                    self.byte_mappings
                        .get(&byte)
                        .expect("expecting mapping for the byte"),
                );
            }

            let part: String = chars.iter().cloned().collect();

            let symbols = self.bpe(part);

            for symbol in &symbols {
                tokens.push(
                    *self
                        .mappings
                        .get(symbol)
                        .expect("expected mapping for the token"),
                );
            }
        }

        tokens
    }

    fn bpe(&self, part: String) -> Vec<String> {
        let mut pairs: HashSet<_> = iter::zip(part.chars(), part.chars().skip(1))
            .map(|(left, right)| Merge(left.to_string(), right.to_string()))
            .collect();

        let mut symbols: Vec<String> = part.chars().map(|c| c.to_string()).collect();

        loop {
            if pairs.is_empty() {
                return symbols;
            }

            let merges_with_priority: Vec<(Merge, u32)> = pairs
                .into_iter()
                .flat_map(|potential_merge| {
                    self.merges
                        .get(&potential_merge)
                        .map(|idx| (potential_merge.clone(), *idx))
                })
                .collect();

            if let Some((merge, _)) = merges_with_priority.iter().min_by_key(|tuple| tuple.1) {
                let left = &merge.0;
                let right = &merge.1;

                let mut new_symbols = Vec::new();

                let mut i = 0;

                while i < symbols.len() {
                    if i == symbols.len() - 1 {
                        new_symbols.push(symbols[i].clone());
                        break;
                    }

                    if &symbols[i] == left && &symbols[i + 1] == right {
                        new_symbols.push(format!("{left}{right}"));
                        i += 2;
                    } else {
                        new_symbols.push(symbols[i].clone());
                        i += 1;
                    }
                }

                symbols = new_symbols;

                if symbols.len() == 1 {
                    break;
                }

                pairs = iter::zip(symbols.iter(), symbols.iter().skip(1))
                    .map(|(left, right)| Merge(left.clone(), right.clone()))
                    .collect();
            } else {
                break;
            }
        }

        symbols
    }

    fn detokenize(&self, input: &[Token]) -> String {
        let mut text = String::new();

        for token in input {
            text.push_str(self.reverse_mappings.get(token).expect("token not found"));
        }

        let mut bytes = Vec::new();

        for char in text.chars() {
            bytes.push(
                *self
                    .reverse_byte_mappings
                    .get(&char)
                    .expect("char mapping not found"),
            );
        }

        String::from_utf8(bytes).expect("must be valid utf-8 after reverting the mapping")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_tokenizes_correctly_1() {
        // given
        let tokenizer = Tokenizer::new();
        let input = "Hello world!";

        // when
        let tokens = tokenizer.tokenize(input);

        // then
        // taken from https://tokenizer.model.box/?model=gpt2
        assert_eq!(tokens, vec![15496, 995, 0]);
    }

    #[test]
    fn it_tokenizes_correctly_2() {
        // given
        let tokenizer = Tokenizer::new();
        let input = "This is something more complex.";

        // when
        let tokens = tokenizer.tokenize(input);

        // then
        // taken from https://tokenizer.model.box/?model=gpt2
        assert_eq!(tokens, vec![1212, 318, 1223, 517, 3716, 13]);
    }

    #[test]
    fn it_tokenizes_correctly_3() {
        // given
        let tokenizer = Tokenizer::new();
        let input = "Eso es algo más complicado.";

        // when
        let tokens = tokenizer.tokenize(input);

        // then
        // taken from https://tokenizer.model.box/?model=gpt2
        assert_eq!(
            tokens,
            vec![36, 568, 1658, 435, 2188, 285, 40138, 2299, 291, 4533, 13]
        );
    }

    #[test]
    fn it_tokenizes_correctly_4() {
        // given
        let tokenizer = Tokenizer::new();
        let input = "これはもっと複雑なものです。";

        // when
        let tokens = tokenizer.tokenize(input);

        // then
        // taken from https://tokenizer.model.box/?model=gpt2
        assert_eq!(
            tokens,
            vec![
                46036, 39258, 31676, 43266, 33180, 30201, 164, 97, 229, 37239, 239, 26945, 43266,
                5641, 30640, 33623, 16764
            ]
        );
    }

    #[test]
    fn it_detokenizes_correctly_1() {
        // given
        let tokenizer = Tokenizer::new();
        let input = vec![15496, 995, 0];

        // when
        let tokens = tokenizer.detokenize(&input);

        // then
        // taken from https://tokenizer.model.box/?model=gpt2
        assert_eq!(tokens, "Hello world!");
    }

    #[test]
    fn it_detokenizes_correctly_2() {
        // given
        let tokenizer = Tokenizer::new();
        let input = vec![1212, 318, 1223, 517, 3716, 13];

        // when
        let tokens = tokenizer.detokenize(&input);

        // then
        // taken from https://tokenizer.model.box/?model=gpt2
        assert_eq!(tokens, "This is something more complex.");
    }

    #[test]
    fn it_detokenizes_correctly_3() {
        // given
        let tokenizer = Tokenizer::new();
        let input = vec![36, 568, 1658, 435, 2188, 285, 40138, 2299, 291, 4533, 13];

        // when
        let tokens = tokenizer.detokenize(&input);

        // then
        // taken from https://tokenizer.model.box/?model=gpt2
        assert_eq!(tokens, "Eso es algo más complicado.");
    }

    #[test]
    fn it_detokenizes_correctly_4() {
        // given
        let tokenizer = Tokenizer::new();
        let input = vec![
            46036, 39258, 31676, 43266, 33180, 30201, 164, 97, 229, 37239, 239, 26945, 43266, 5641,
            30640, 33623, 16764,
        ];

        // when
        let tokens = tokenizer.detokenize(&input);

        // then
        // taken from https://tokenizer.model.box/?model=gpt2
        assert_eq!(tokens, "これはもっと複雑なものです。");
    }

    #[test]
    fn it_tokenizes_and_detokenizes_correctly() {
        // given
        let tokenizer = Tokenizer::new();
        let input = "This is an end-to-end integration";

        // when
        let output = tokenizer.detokenize(&tokenizer.tokenize(input));

        // then
        assert_eq!(input, output);
    }

    #[test]
    fn it_tokenizes_and_detokenizes_utf8_correctly() {
        // given
        let tokenizer = Tokenizer::new();
        let input = "これはエンドツーエンドの統合です。";

        // when
        let output = tokenizer.detokenize(&tokenizer.tokenize(input));

        // then
        assert_eq!(input, output);
    }
}
