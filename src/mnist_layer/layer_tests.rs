use crate::mnist_layer::{Aesir3Layer, MNIST_AREA};
use mnist::{Mnist, MnistBuilder};
use std::convert::TryInto;

#[test]
fn test_if_all_synapses_formed() {
    let num_neurons = 100;

    let aesir3_layer = Aesir3Layer::new(
        Vec::new(),
        Vec::new(),
        num_neurons,
        10.0,
        8.,
        0.1,
        Box::new(|| 0.1),
        Box::new(|| 0.1),
    );

    for neuron in aesir3_layer.neurons.iter() {
        assert_eq!(
            neuron.get_synapse_count(),
            MNIST_AREA + num_neurons as usize - 1
        );
    }
}

/// This test actually doesn't assert anything
/// It's just for me to look at the actual rendered
/// images and make sure they're rendering correctly
#[test]
fn test_display_img() {
    let Mnist {
        trn_img, trn_lbl, ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(2000)
        .validation_set_length(200)
        .test_set_length(200)
        .finalize();

    let trn_img: Vec<f32> = trn_img.iter().map(|val| *val as f32 / 255.).collect();

    let aesir3_layer = Aesir3Layer::new(
        trn_img,
        trn_lbl,
        100,
        10.0,
        8.,
        0.1,
        Box::new(|| 0.1),
        Box::new(|| 0.1),
    );

    aesir3_layer.display_img(50);
    aesir3_layer.display_img(201);
    aesir3_layer.display_img(1047);
    aesir3_layer.display_img(1048);
    aesir3_layer.display_img(1049);
    aesir3_layer.display_img(1050);
}
