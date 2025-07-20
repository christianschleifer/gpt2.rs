fn main() {
    println!("Hello, world!");
}

struct Tokenizer {}

impl Tokenizer {
    fn new() -> Self {
        todo!()
    }

    fn tokenize(&self, input: &str) -> Vec<u32> {
        todo!()
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
}
