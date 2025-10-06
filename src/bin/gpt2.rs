use regex::Regex;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};

use std::error::Error;
use std::mem;
use std::path::Path;
use std::{fs, iter};

// see https://openaipublic.blob.core.windows.net/gpt-2/models/124M/hparams.json
const NUM_LAYERS: u8 = 12;

fn main() {
    let tokenizer = Tokenizer::new();

    let input_string = "Hello world!";

    println!("Tokenizing the input string {input_string}");
    let tokens = tokenizer.tokenize(input_string);
    println!("Got tokens: {tokens:?}");

    let mut llm = LargeLanguageModel::try_from(Path::new("resources/model.safetensors")).unwrap();

    llm.initialize(&tokens[..tokens.len() - 1]);
    println!("Initialized the LLM");

    let num_samples: u32 = 2;

    println!("Sampling {num_samples} tokens.");
    let mut curr_token = *tokens.last().unwrap();
    for _ in 0..num_samples {
        let next_token = llm.sample_one(curr_token);
        curr_token = next_token;

        let next_output = tokenizer.detokenize(&vec![next_token]);
        print!("{next_output}");
    }
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

#[derive(Deserialize, Debug)]
struct LayerSpec {
    shape: Vec<u64>,
    data_offsets: Vec<u64>,
}

#[derive(Deserialize, Debug)]
struct HeaderSpec {
    #[serde(rename = "__metadata__")]
    _metadata: serde_json::Value,
    #[serde(flatten)]
    layers: HashMap<String, LayerSpec>,
}

struct SafeTensors<'a> {
    layers: HashMap<String, LayerSpec>,
    raw_data: &'a [u8],
}

impl<'a> SafeTensors<'a> {
    fn new(layers: HashMap<String, LayerSpec>, raw_data: &'a [u8]) -> Self {
        Self { layers, raw_data }
    }
}

impl<'a> SafeTensors<'a> {
    fn get_tensor_by_name(&self, name: &str) -> Tensor2D {
        let layer_def = self
            .layers
            .get(name)
            .unwrap_or_else(|| panic!("must contain tensor with name {name}"));

        let raw_data =
            &self.raw_data[layer_def.data_offsets[0] as usize..layer_def.data_offsets[1] as usize];

        let mut data = Vec::with_capacity(raw_data.len() / mem::size_of::<f32>());

        for raw_float in raw_data.chunks_exact(mem::size_of::<f32>()) {
            let float = f32::from_le_bytes(
                raw_float
                    .try_into()
                    .expect("data must encode valid f32 values"),
            );
            data.push(float);
        }

        let mut shape_iter = layer_def.shape.iter();

        let shape = (
            *shape_iter.next().unwrap() as usize,
            *shape_iter.next().unwrap_or(&1) as usize,
        );

        Tensor2D::new(shape, data)
    }
}

#[derive(Debug)]
enum Dimension {
    Row,
    Column,
}

#[derive(Debug)]
struct Tensor2D {
    shape: (usize, usize),
    data: Vec<f32>,
}

impl Tensor2D {
    fn new(shape: (usize, usize), data: Vec<f32>) -> Self {
        Self { shape, data }
    }

    fn get(&self, dim: Dimension, index: usize) -> &[f32] {
        match dim {
            Dimension::Row => {
                let real_index = index * self.shape.1;
                &self.data[real_index..real_index + self.shape.1]
            }
            Dimension::Column => {
                let real_index = index * self.shape.0;
                &self.data[real_index..real_index + self.shape.0]
            }
        }
    }
}

#[derive(Debug)]
struct TransformerBlock {
    layer_norm_1_weight: Tensor2D, // (n_embd,)
    layer_norm_1_bias: Tensor2D,   // (n_embd, )

    layer_norm_2_weight: Tensor2D, // (n_embd, )
    layer_norm_2_bias: Tensor2D,   // (n_embd, )

    mlp_fc_weight: Tensor2D, // (n_embd, n_neurons)
    mlp_fc_bias: Tensor2D,   // (n_neurons, )

    mlp_proj_weight: Tensor2D, // (n_neurons, n_embd)
    mlp_proj_bias: Tensor2D,   // (n_embd, )
}

impl TransformerBlock {
    fn run(&self, context_vector: Vec<f32>) -> Vec<f32> {
        todo!()
    }
}

#[derive(Debug)]
struct LargeLanguageModel {
    wte: Tensor2D, // (n_vocab, n_embd)
    wpe: Tensor2D, // (n_ctx, n_embd)

    transformer_blocks: Vec<TransformerBlock>,

    current_position: usize,
}

impl LargeLanguageModel {
    fn new(wte: Tensor2D, wpe: Tensor2D, transformer_blocks: Vec<TransformerBlock>) -> Self {
        Self {
            wte,
            wpe,
            transformer_blocks,
            current_position: 0,
        }
    }

    fn initialize(&mut self, tokens: &[Token]) {
        for token in tokens {
            self.sample_one(*token);
        }
    }

    fn sample_one(&mut self, token: Token) -> Token {
        let token_embedding = self.wte.get(Dimension::Row, token as usize);
        let position_embedding = self.wpe.get(Dimension::Row, self.current_position);

        let mut context_vector: Vec<f32> = token_embedding
            .iter()
            .zip(position_embedding.iter())
            .map(|(left, right)| left + right)
            .collect();

        for transformer_block in self.transformer_blocks.iter_mut() {
            context_vector = transformer_block.run(context_vector);
        }

        println!("{context_vector:?}");
        self.current_position += 1;
        0
    }
}

impl TryFrom<&Path> for LargeLanguageModel {
    type Error = Box<dyn Error>;

    fn try_from(path: &Path) -> Result<Self, Self::Error> {
        let bytes = fs::read(path)?;

        let bytes_for_header_length = mem::size_of::<u64>();

        let header_size = u64::from_le_bytes(bytes[..bytes_for_header_length].try_into()?);

        let header_data =
            &bytes[bytes_for_header_length..(header_size as usize + bytes_for_header_length)];

        let header: HeaderSpec = serde_json::from_slice(header_data)?;

        let safe_tensors = SafeTensors::new(
            header.layers,
            &bytes[bytes_for_header_length + header_size as usize..],
        );

        let wte = safe_tensors.get_tensor_by_name("wte.weight");
        let wpe = safe_tensors.get_tensor_by_name("wpe.weight");

        for i in 0..NUM_LAYERS {}

        Ok(LargeLanguageModel::new(wte, wpe, vec![]))
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
