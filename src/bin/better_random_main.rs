use mnist::{Mnist, MnistBuilder};
use aesir3_mnist::better_random_layer::RandomDealLayer;

const NUM_NEURONS: usize = 400;
const MIN_WEIGHT: f32 = -10.0;
const MAX_WEIGHT: f32 = 10.0;

const TRAINING_SET_LENGTH: u32 = 40_000;
const TEST_SET_LENGTH: u32 = 10_000;

fn main() {
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
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

    let random_layer = RandomDealLayer::new(
        train_img,
        trn_lbl,
        NUM_NEURONS,
        MIN_WEIGHT,
        MAX_WEIGHT
    );

    let accuracy = random_layer.get_classification_accuracy(test_img, tst_lbl);

    println!("Model Accuracy: {}", accuracy);
}
