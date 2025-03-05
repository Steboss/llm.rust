use clap::{App, Arg};

pub struct Config {
    pub learning_rate: f32,
    pub batch_size: usize,
    pub epochs: usize,
}

impl Config {
    pub fn from_args() -> Self {
        let matches = App::new("llm")
            .version("0.1.0")
            .author("Adesoji Alu <you@example.com>")
            .about("Rust LLM Training")
            .arg(
                Arg::with_name("learning_rate")
                    .short("l")
                    .long("learning_rate")
                    .value_name("FLOAT")
                    .help("Sets the learning rate")
                    .takes_value(true),
            )
            .arg(
                Arg::with_name("batch_size")
                    .short("b")
                    .long("batch_size")
                    .value_name("INT")
                    .help("Sets the batch size")
                    .takes_value(true),
            )
            .arg(
                Arg::with_name("epochs")
                    .short("e")
                    .long("epochs")
                    .value_name("INT")
                    .help("Sets the number of epochs")
                    .takes_value(true),
            )
            .get_matches();

        let learning_rate = matches
            .value_of("learning_rate")
            .unwrap_or("0.001")
            .parse()
            .expect("Invalid learning rate");
        let batch_size = matches
            .value_of("batch_size")
            .unwrap_or("32")
            .parse()
            .expect("Invalid batch size");
        let epochs = matches
            .value_of("epochs")
            .unwrap_or("10")
            .parse()
            .expect("Invalid number of epochs");

        Config {
            learning_rate,
            batch_size,
            epochs,
        }
    }
}
