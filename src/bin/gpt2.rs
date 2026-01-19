use regex::Regex;

use serde::Deserialize;
use std::collections::{HashMap, HashSet};

use std::error::Error;
use std::io::{self, BufWriter, Write};
use std::mem;
use std::ops::{AddAssign, DivAssign, MulAssign};
use std::path::Path;
use std::{fs, iter};

// see https://openaipublic.blob.core.windows.net/gpt-2/models/124M/hparams.json
const NUM_LAYERS: u8 = 12;
const NUM_HEADS: u8 = 12;

fn main() {
    let tokenizer = Tokenizer::new();

    let input_string = "Medicine is ";

    let tokens = tokenizer.tokenize(input_string);

    let mut llm = LargeLanguageModel::try_from(Path::new("resources/model.safetensors")).unwrap();

    llm.initialize(&tokens[..tokens.len() - 1]);

    let num_samples: u32 = 25;

    let mut writer = BufWriter::new(io::stdout());
    print!("{input_string}");
    let mut curr_token = *tokens.last().unwrap();
    for _ in 0..num_samples {
        let next_token = llm.sample_one(curr_token);
        curr_token = next_token;

        let next_output = tokenizer.detokenize(&[next_token]);
        write!(writer, "{next_output}").unwrap();
        writer.flush().unwrap();
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

        let shape = if layer_def.shape.len() == 1 {
            (1, layer_def.shape[0] as usize)
        } else {
            (layer_def.shape[0] as usize, layer_def.shape[1] as usize)
        };

        Tensor2D::new(shape, data)
    }
}

#[derive(Debug)]
enum Dimension {
    Row,
    Column,
}

#[derive(Clone, Debug)]
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

    fn mat_mul(&self, other: &Tensor2D) -> Tensor2D {
        if self.shape.1 != other.shape.0 {
            panic!("invalid shapes for matrix multiplication");
        }

        let mut data = vec![0_f32; self.shape.0 * other.shape.1];

        for row_idx_a in 0..self.shape.0 {
            for col_idx_b in 0..other.shape.1 {
                let mut sum = 0_f32;
                for col_idx_a in 0..self.shape.1 {
                    sum += self.data[row_idx_a * self.shape.1 + col_idx_a]
                        * other.data[col_idx_a * other.shape.1 + col_idx_b];
                }

                data[row_idx_a * other.shape.1 + col_idx_b] = sum;
            }
        }

        Tensor2D::new((self.shape.0, other.shape.1), data)
    }

    fn standardize(&mut self) {
        // normalization
        let len = self.data.len() as f32;
        let mean: f32 = self.data.iter().sum::<f32>() / len;
        let variance: f32 = self
            .data
            .iter()
            .map(|val| (val - mean).powi(2))
            .sum::<f32>()
            / len;

        let epsilon = 1e-5;
        let std_dev = (variance + epsilon).sqrt();

        for i in 0..self.data.len() {
            self.data[i] = (self.data[i] - mean) / std_dev;
        }
    }

    fn soft_max(&mut self) {
        let max = self.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let mut sum = 0_f32;

        for val in &mut self.data {
            *val = (*val - max).exp();
            sum += *val;
        }

        for val in &mut self.data {
            *val /= sum;
        }
    }

    fn gelu(&mut self) {
        const SQRT_2_OVER_PI: f32 = 0.797_884_6;

        for val in &mut self.data {
            *val = 0.5 * *val * (1.0 + (SQRT_2_OVER_PI * (*val + 0.044_715 * val.powi(3))).tanh());
        }
    }

    fn transpose(&self) -> Tensor2D {
        let (rows, cols) = self.shape;
        let mut data = vec![0_f32; rows * cols];

        for row in 0..rows {
            for col in 0..cols {
                data[col * rows + row] = self.data[row * cols + col];
            }
        }

        Tensor2D::new((cols, rows), data)
    }
}

impl AddAssign<&Tensor2D> for Tensor2D {
    fn add_assign(&mut self, rhs: &Self) {
        if !(self.shape.0 == rhs.shape.0 && self.shape.1 == rhs.shape.1) {
            panic!("tensors must be of same shape for addition");
        }
        for row_idx in 0..self.shape.0 {
            for col_idx in 0..self.shape.1 {
                let real_idx = row_idx * self.shape.1 + col_idx;

                self.data[real_idx] += rhs.data[real_idx];
            }
        }
    }
}

impl MulAssign<&Tensor2D> for Tensor2D {
    fn mul_assign(&mut self, rhs: &Self) {
        if !(self.shape.0 == rhs.shape.0 && self.shape.1 == rhs.shape.1) {
            panic!("tensors must be of same shape for multiplication");
        }
        for row_idx in 0..self.shape.0 {
            for col_idx in 0..self.shape.1 {
                let real_idx = row_idx * self.shape.1 + col_idx;

                self.data[real_idx] *= rhs.data[real_idx];
            }
        }
    }
}

impl DivAssign<u8> for Tensor2D {
    fn div_assign(&mut self, rhs: u8) {
        for row_idx in 0..self.shape.0 {
            for col_idx in 0..self.shape.1 {
                let real_index = row_idx * self.shape.1 + col_idx;
                self.data[real_index] /= rhs as f32;
            }
        }
    }
}

#[derive(Debug)]
struct TransformerBlock {
    ln_1_weight: Tensor2D, // (n_embd,)
    ln_1_bias: Tensor2D,   // (n_embd, )

    attn_c_attn_weight: Tensor2D, // (n_embd, 3 * n_embd)
    attn_c_attn_bias: Tensor2D,   // (3 * n_embd, )
    //
    attn_c_proj_weight: Tensor2D, // (n_embd, n_embd)
    attn_c_proj_bias: Tensor2D,   // (1, n_embd)

    ln_2_weight: Tensor2D, // (n_embd, )
    ln_2_bias: Tensor2D,   // (n_embd, )

    mlp_fc_weight: Tensor2D, // (n_embd, n_neurons)
    mlp_fc_bias: Tensor2D,   // (n_neurons, )

    mlp_proj_weight: Tensor2D, // (n_neurons, n_embd)
    mlp_proj_bias: Tensor2D,   // (n_embd, )

    key_cache: Vec<Tensor2D>,
    value_cache: Vec<Tensor2D>,
}

impl TransformerBlock {
    #[allow(clippy::too_many_arguments)]
    fn new(
        ln_1_weight: Tensor2D,
        ln_1_bias: Tensor2D,
        attn_c_attn_weight: Tensor2D,
        attn_c_attn_bias: Tensor2D,
        attn_c_proj_weight: Tensor2D,
        attn_c_proj_bias: Tensor2D,
        ln_2_weight: Tensor2D,
        ln_2_bias: Tensor2D,
        mlp_fc_weight: Tensor2D,
        mlp_fc_bias: Tensor2D,
        mlp_proj_weight: Tensor2D,
        mlp_proj_bias: Tensor2D,
    ) -> Self {
        Self {
            ln_1_weight,
            ln_1_bias,
            attn_c_attn_weight,
            attn_c_attn_bias,
            attn_c_proj_weight,
            attn_c_proj_bias,
            ln_2_weight,
            ln_2_bias,
            mlp_fc_weight,
            mlp_fc_bias,
            mlp_proj_weight,
            mlp_proj_bias,
            key_cache: vec![],
            value_cache: vec![],
        }
    }

    fn run(&mut self, mut context_tensor: Tensor2D) -> Tensor2D {
        let residual = context_tensor.clone();

        // layer norm 1
        context_tensor.standardize();
        context_tensor *= &self.ln_1_weight;
        context_tensor += &self.ln_1_bias;

        // multi-head attention
        context_tensor = context_tensor.mat_mul(&self.attn_c_attn_weight);
        context_tensor += &self.attn_c_attn_bias;

        let n_embd = context_tensor.shape.1 / 3; // must contain query, key and value

        let query = Tensor2D::new((1, n_embd), context_tensor.data[0..n_embd].to_owned());

        // add key and value vector to the cache
        self.key_cache.push(Tensor2D::new(
            (1, n_embd),
            context_tensor.data[n_embd..n_embd * 2].to_owned(),
        ));
        self.value_cache.push(Tensor2D::new(
            (1, n_embd),
            context_tensor.data[n_embd * 2..n_embd * 3].to_owned(),
        ));

        let head_dim = n_embd / NUM_HEADS as usize;

        let mut all_attention_scores = Vec::with_capacity(n_embd);

        for head_idx in 0..NUM_HEADS as usize {
            let start_idx = head_idx * head_dim;
            let end_idx = start_idx + head_dim;

            let query = Tensor2D::new((1, head_dim), query.data[start_idx..end_idx].to_owned());

            let mut key_data = Vec::with_capacity(head_dim * self.key_cache.len());
            let mut value_data = Vec::with_capacity(head_dim * self.key_cache.len());

            for i in 0..self.key_cache.len() {
                key_data.extend_from_slice(&self.key_cache[i].data[start_idx..end_idx]);
                value_data.extend_from_slice(&self.value_cache[i].data[start_idx..end_idx]);
            }

            let key = Tensor2D::new((self.key_cache.len(), head_dim), key_data);
            let value = Tensor2D::new((self.key_cache.len(), head_dim), value_data);

            let mut attention_scores = query.mat_mul(&key.transpose()); // (1, n_pos)
            attention_scores /= 8; // sqrt of head_dim
            attention_scores.soft_max();

            attention_scores = attention_scores.mat_mul(&value); // (1, head_dim)
            all_attention_scores.extend_from_slice(attention_scores.get(Dimension::Row, 0));
        }

        let mut context_tensor = Tensor2D::new((1, n_embd), all_attention_scores);
        context_tensor = context_tensor.mat_mul(&self.attn_c_proj_weight);
        context_tensor += &self.attn_c_proj_bias;
        context_tensor += &residual;

        let residual = context_tensor.clone();

        // layer norm 2
        context_tensor.standardize();
        context_tensor *= &self.ln_2_weight;
        context_tensor += &self.ln_2_bias;

        context_tensor = context_tensor.mat_mul(&self.mlp_fc_weight);
        context_tensor += &self.mlp_fc_bias;

        context_tensor.gelu();

        context_tensor = context_tensor.mat_mul(&self.mlp_proj_weight);
        context_tensor += &self.mlp_proj_bias;

        context_tensor += &residual;

        context_tensor
    }
}

#[derive(Debug)]
struct LargeLanguageModel {
    wte: Tensor2D, // (n_vocab, n_embd)
    wpe: Tensor2D, // (n_ctx, n_embd)

    transformer_blocks: Vec<TransformerBlock>,

    ln_f_weight: Tensor2D, // (n_embd,)
    ln_f_bias: Tensor2D,   // (n_embd,)

    current_position: usize,
}

impl LargeLanguageModel {
    fn new(
        wte: Tensor2D,
        wpe: Tensor2D,
        transformer_blocks: Vec<TransformerBlock>,
        ln_f_weight: Tensor2D,
        ln_f_bias: Tensor2D,
    ) -> Self {
        Self {
            wte,
            wpe,
            transformer_blocks,
            ln_f_weight,
            ln_f_bias,
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

        // len == n_embd
        let context_vector: Vec<f32> = token_embedding
            .iter()
            .zip(position_embedding.iter())
            .map(|(left, right)| left + right)
            .collect();

        let mut context_tensor = Tensor2D::new((1, context_vector.len()), context_vector);

        for transformer_block in self.transformer_blocks.iter_mut() {
            context_tensor = transformer_block.run(context_tensor);
        }

        context_tensor.standardize();
        context_tensor *= &self.ln_f_weight;
        context_tensor += &self.ln_f_bias;

        let logits = context_tensor.mat_mul(&self.wte.transpose());

        let token = logits
            .data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx as Token)
            .unwrap();

        self.current_position += 1;
        token
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

        let mut transformer_blocks: Vec<TransformerBlock> = Vec::new();

        for i in 0..NUM_LAYERS {
            let ln_1_weight = safe_tensors.get_tensor_by_name(&format!("h.{i}.ln_1.weight"));
            let ln_1_bias = safe_tensors.get_tensor_by_name(&format!("h.{i}.ln_1.bias"));

            let attn_c_attn_weight =
                safe_tensors.get_tensor_by_name(&format!("h.{i}.attn.c_attn.weight"));
            let attn_c_attn_bias =
                safe_tensors.get_tensor_by_name(&format!("h.{i}.attn.c_attn.bias"));

            let attn_c_proj_weight =
                safe_tensors.get_tensor_by_name(&format!("h.{i}.attn.c_proj.weight"));
            let attn_c_proj_bias =
                safe_tensors.get_tensor_by_name(&format!("h.{i}.attn.c_proj.bias"));

            let ln_2_weight = safe_tensors.get_tensor_by_name(&format!("h.{i}.ln_2.weight"));
            let ln_2_bias = safe_tensors.get_tensor_by_name(&format!("h.{i}.ln_2.bias"));

            let mlp_c_fc_weight =
                safe_tensors.get_tensor_by_name(&format!("h.{i}.mlp.c_fc.weight"));
            let mlp_c_fc_bias = safe_tensors.get_tensor_by_name(&format!("h.{i}.mlp.c_fc.bias"));

            let mlp_c_proj_weight =
                safe_tensors.get_tensor_by_name(&format!("h.{i}.mlp.c_proj.weight"));
            let mlp_c_proj_bias =
                safe_tensors.get_tensor_by_name(&format!("h.{i}.mlp.c_proj.bias"));

            let transformer_block = TransformerBlock::new(
                ln_1_weight,
                ln_1_bias,
                attn_c_attn_weight,
                attn_c_attn_bias,
                attn_c_proj_weight,
                attn_c_proj_bias,
                ln_2_weight,
                ln_2_bias,
                mlp_c_fc_weight,
                mlp_c_fc_bias,
                mlp_c_proj_weight,
                mlp_c_proj_bias,
            );

            transformer_blocks.push(transformer_block);
        }

        let ln_f_weight = safe_tensors.get_tensor_by_name("ln_f.weight");
        let ln_f_bias = safe_tensors.get_tensor_by_name("ln_f.bias");

        Ok(LargeLanguageModel::new(
            wte,
            wpe,
            transformer_blocks,
            ln_f_weight,
            ln_f_bias,
        ))
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

    #[test]
    fn mat_mul_2x3_times_3x2() {
        // given
        let a = Tensor2D::new((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Tensor2D::new((3, 2), vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);

        // when
        let c = a.mat_mul(&b);

        // then
        // [1,2,3] * [7,9,11] = 7+18+33 = 58    [1,2,3] * [8,10,12] = 8+20+36 = 64
        // [4,5,6] * [7,9,11] = 28+45+66 = 139  [4,5,6] * [8,10,12] = 32+50+72 = 154
        assert_eq!(c.shape, (2, 2));
        assert_eq!(c.data, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn mat_mul_with_identity() {
        // given
        let a = Tensor2D::new((2, 2), vec![5.0, 3.0, 2.0, 7.0]);
        let identity = Tensor2D::new((2, 2), vec![1.0, 0.0, 0.0, 1.0]);

        // when
        let c = a.mat_mul(&identity);

        // then
        assert_eq!(c.shape, (2, 2));
        assert_eq!(c.data, vec![5.0, 3.0, 2.0, 7.0]);
    }

    #[test]
    fn mat_mul_different_output_shape() {
        // given
        let a = Tensor2D::new((2, 2), vec![1.0, 2.0, 3.0, 4.0]);
        let b = Tensor2D::new((2, 3), vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

        // when
        let c = a.mat_mul(&b);

        // then
        // [1,2] * [5,8] = 5+16 = 21   [1,2] * [6,9] = 6+18 = 24   [1,2] * [7,10] = 7+20 = 27
        // [3,4] * [5,8] = 15+32 = 47  [3,4] * [6,9] = 18+36 = 54  [3,4] * [7,10] = 21+40 = 61
        assert_eq!(c.shape, (2, 3));
        assert_eq!(c.data, vec![21.0, 24.0, 27.0, 47.0, 54.0, 61.0]);
    }

    #[test]
    #[should_panic(expected = "invalid shapes for matrix multiplication")]
    fn mat_mul_incompatible_shapes_panics() {
        // given
        let a = Tensor2D::new((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Tensor2D::new((2, 2), vec![1.0, 2.0, 3.0, 4.0]);

        // when
        a.mat_mul(&b);
    }

    #[test]
    fn add_assign_element_wise() {
        // given
        let mut a = Tensor2D::new((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Tensor2D::new((2, 3), vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);

        // when
        a += &b;

        // then
        assert_eq!(a.shape, (2, 3));
        assert_eq!(a.data, vec![11.0, 22.0, 33.0, 44.0, 55.0, 66.0]);
    }

    #[test]
    #[should_panic(expected = "tensors must be of same shape for addition")]
    fn add_assign_incompatible_shapes_panics() {
        // given
        let mut a = Tensor2D::new((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Tensor2D::new((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // when
        a += &b;
    }

    #[test]
    fn mul_assign_element_wise() {
        // given
        let mut a = Tensor2D::new((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Tensor2D::new((2, 3), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

        // when
        a *= &b;

        // then
        assert_eq!(a.shape, (2, 3));
        assert_eq!(a.data, vec![2.0, 6.0, 12.0, 20.0, 30.0, 42.0]);
    }

    #[test]
    #[should_panic(expected = "tensors must be of same shape for multiplication")]
    fn mul_assign_incompatible_shapes_panics() {
        // given
        let mut a = Tensor2D::new((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Tensor2D::new((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // when
        a *= &b;
    }

    #[test]
    fn transpose_swaps_rows_and_columns() {
        // given
        // [1, 2, 3]
        // [4, 5, 6]
        let a = Tensor2D::new((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // when
        let t = a.transpose();

        // then
        // [1, 4]
        // [2, 5]
        // [3, 6]
        assert_eq!(t.shape, (3, 2));
        assert_eq!(t.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }
}
