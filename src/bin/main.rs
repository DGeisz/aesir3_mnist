use mnist::{Mnist, MnistBuilder};
use rand::Rng;
use aesir3_mnist::mnist_layer::Aesir3Layer;

const TRAINING_SET_LENGTH: u32 = 1000;
const TEST_SET_LENGTH: u32 = 2000;

const NUM_NEURONS: u32 = 100;
const FIRE_THRESHOLD: f32 = 10.0;
const MAX_SYNAPSE_WEIGHT: f32 = 100.;
const LEARNING_CONSTANT: f32 = 0.001;

const NUM_CYCLES: u32 = 5;

const SENS_SYN_START_MAX_WEIGHT: f32 = 0.3;
const NEURON_SYN_START_MAX_WEIGHT: f32 = 0.3;

fn main() {
    let sensor_synapse_weight_gen =
        Box::new(|| rand::thread_rng().gen_range(0., SENS_SYN_START_MAX_WEIGHT));
    let neuron_synapse_weight_gen =
        Box::new(|| rand::thread_rng().gen_range(0.0, NEURON_SYN_START_MAX_WEIGHT));

    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        // tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(TRAINING_SET_LENGTH)
        .validation_set_length(TEST_SET_LENGTH)
        .test_set_length(TEST_SET_LENGTH)
        .finalize();

    let mut train_img = Vec::new();

    for val in trn_img {
        train_img.push(val as f32 / 255.0);
    }

    let mut test_img = Vec::new();

    for val in tst_img {
        test_img.push(val as f32 / 255.0);
    }

    let aesir3_layer = Aesir3Layer::new(
        train_img,
        trn_lbl,
        NUM_NEURONS,
        FIRE_THRESHOLD,
        MAX_SYNAPSE_WEIGHT,
        LEARNING_CONSTANT,
        sensor_synapse_weight_gen,
        neuron_synapse_weight_gen,
    );

    aesir3_layer.train_layer(NUM_CYCLES);

    // let classifier = aesir3_layer.gen_classifier(NUM_CYCLES);
    //
    // let accuracy = aesir3_layer.get_classifier_accuracy(
    //     test_img,
    //     tst_lbl,
    //     &classifier,
    //     NUM_CYCLES
    // );
    //
    // print!("Model accuracy: {}", accuracy);
}
