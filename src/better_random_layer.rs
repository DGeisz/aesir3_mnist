use rand::Rng;
use std::cell::RefCell;

pub const MNIST_SIDE_LENGTH: usize = 28;
pub const MNIST_AREA: usize = MNIST_SIDE_LENGTH * MNIST_SIDE_LENGTH;

pub struct RandomNeuron {
    synapses: Vec<f32>,
    internal_charge: RefCell<f32>,
}

impl RandomNeuron {
    pub fn new(synapses: Vec<f32>) -> RandomNeuron {
        RandomNeuron {
            synapses,
            internal_charge: RefCell::new(0.0),
        }
    }

    pub fn incr(&self, measure: f32) {
        *self.internal_charge.borrow_mut() += measure;
    }

    pub fn clear(&self) {
        *self.internal_charge.borrow_mut() = 0.0;
    }

    pub fn get_charge(&self) -> f32 {
        *self.internal_charge.borrow()
    }
}

pub struct RandomDealLayerClassifier {
    neuron_class: Vec<u8>,
    neuron_class_weight: Vec<f32>,
    class_neuron_total_weight: [f32; 10],
}

pub struct RandomDealLayer {
    neurons: Vec<RandomNeuron>,
    classifier: RandomDealLayerClassifier,
}

impl RandomDealLayer {
    pub fn new(
        train_img_vec: Vec<f32>, // This should already be modified so values are between 0 and 1
        train_lbl_vec: Vec<u8>,

        num_neurons: usize,

        min_weight: f32,
        max_weight: f32,
    ) -> RandomDealLayer {
        let mut neurons = Vec::new();

        for _ in 0..num_neurons {
            let mut synapses = Vec::new();

            for _ in 0..MNIST_AREA {
                synapses.push(rand::thread_rng().gen_range(min_weight, max_weight));
            }

            neurons.push(RandomNeuron::new(synapses));
        }

        let mut neuron_class_measures = Vec::new();

        for _ in 0..num_neurons {
            neuron_class_measures.push([0.0; 10]);
        }

        for img_index in 0..train_lbl_vec.len() {
            // Clear neurons
            for neuron in &neurons {
                neuron.clear();
            }

            // Run past neurons
            for (i, img_i) in ((img_index * MNIST_AREA)..((img_index + 1) * MNIST_AREA)).enumerate()
            {
                for neuron in &neurons {
                    neuron
                        .incr(*neuron.synapses.get(i).unwrap() * train_img_vec.get(img_i).unwrap());
                }
            }

            let label = *train_lbl_vec.get(img_index).unwrap() as usize;
            for (class_measure, neuron) in neuron_class_measures.iter_mut().zip(neurons.iter()) {
                class_measure[label] += neuron.get_charge();
            }

            if img_index % 1000 == 0 {
                println!("Labeled img: {}", img_index);
            }
        }

        //Clear neurons once more
        for neuron in &neurons {
            neuron.clear();
        }

        let mut neuron_classes = Vec::new();
        let mut neuron_class_weights = Vec::new();
        let mut neuron_class_total_weights = [0.0_f32; 10];

        for class_measure in &neuron_class_measures {
            let mut max_index = 0;
            let mut max_measure = class_measure[0];

            let mut total_measure = class_measure[0];

            for i in 1..10 {
                if class_measure[i] > max_measure {
                    max_measure = class_measure[i];
                    max_index = i as u8;
                }

                total_measure += class_measure[i];
            }

            let weight = (max_measure - (total_measure / 10.)) / (total_measure / 10.).abs();

            neuron_classes.push(max_index);
            neuron_class_weights.push(weight);
            neuron_class_total_weights[max_index as usize] += weight;
        }


        let classifier = RandomDealLayerClassifier {
            neuron_class: neuron_classes,
            neuron_class_weight: neuron_class_weights,
            class_neuron_total_weight: neuron_class_total_weights
        };

        RandomDealLayer {
            neurons,
            classifier
        }


        // let mut neuron_classes = Vec::new();
        // let mut class_count = [0_u32; 10];
        //
        // for class_measure in neuron_class_measures.iter() {
        //     let mut max_index = 0;
        //     let mut max_measure = class_measure[0];
        //
        //     for i in 1..10 {
        //         if class_measure[i] > max_measure {
        //             max_measure = class_measure[i];
        //             max_index = i as u8;
        //         }
        //     }
        //
        //     neuron_classes.push(max_index);
        //     class_count[max_index as usize] += 1;
        // }
        //
        // let layer_classifier = RandomLayerClassifier {
        //     neuron_class: neuron_classes,
        //     class_neuron_count: class_count,
        // };
        //
        // RandomDiehlLayer {
        //     neurons,
        //     classifier: layer_classifier,
        // }
        // unimplemented!()
    }

    pub fn classify_img(&self, img_vec: &Vec<f32>, img_index: usize) -> u8 {
        // Clear neurons
        for neuron in &self.neurons {
            neuron.clear();
        }

        for (i, img_i) in ((img_index * MNIST_AREA)..((img_index + 1) * MNIST_AREA)).enumerate()
        {
            for neuron in &self.neurons {
                neuron
                    .incr(*neuron.synapses.get(i).unwrap() * img_vec.get(img_i).unwrap());
            }
        }

        let mut total_class_measure = [0.0; 10];

        for (neuron, (class, weight)) in self.neurons.iter().zip(self.classifier.neuron_class.iter().zip(self.classifier.neuron_class_weight.iter())) {
            total_class_measure[*class as usize] += neuron.get_charge() * weight;
        }

        let mut max_index = 0;
        let mut max_average_measure = total_class_measure[0] / self.classifier.class_neuron_total_weight[0] as f32;

        for i in 1..10 {
            let average_measure = total_class_measure[i] / self.classifier.class_neuron_total_weight[i] as f32;

            if average_measure > max_average_measure {
                max_average_measure = average_measure;
                max_index = i;
            }
        }

        max_index as u8
    }

    pub fn get_classification_accuracy(&self, test_images: Vec<f32>, test_labels: Vec<u8>) -> f32 {
        let mut num_correct = 0.;

        for img_index in 0..test_labels.len() {
            let classification = self.classify_img(&test_images, img_index);

            if classification == *test_labels.get(img_index).unwrap() {
                num_correct += 1.;
            }

            if img_index % 1000 == 0 {
                println!("Finished classification: {}", img_index);
            }
        }

        num_correct / test_labels.len() as f32
    }
}
