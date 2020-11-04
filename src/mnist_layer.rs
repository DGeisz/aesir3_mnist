use aesir3::neuron::{ChargeCycle, Neuronic, NeuronicInput, SynapticType};
use aesir3::{Neuron, NeuronicSensor};
use rulinalg::matrix::Matrix;
use std::rc::Rc;

pub const MNIST_SIDE_LENGTH: usize = 28;
pub const MNIST_AREA: usize = MNIST_SIDE_LENGTH * MNIST_SIDE_LENGTH;

pub struct Aesir3Layer {
    neurons: Vec<Rc<Neuron>>,
    sensors: Vec<Rc<NeuronicSensor>>,
    train_img_vec: Vec<f32>,
    train_lbl_vec: Vec<u8>,
}

impl Aesir3Layer {
    pub fn new(
        train_img_vec: Vec<f32>, // This should already be modified so values are between 0 and 1
        train_lbl_vec: Vec<u8>,

        // Neuron parameters
        num_neurons: u32,
        fire_threshold: f32,
        max_synapse_weight: f32,
        learning_constant: f32,
        sensor_synapse_weight_generator: Box<dyn Fn() -> f32>,
        neuron_synapse_weight_generator: Box<dyn Fn() -> f32>,
    ) -> Aesir3Layer {
        //Create neuron vector
        let sqrt_num_neurons = (num_neurons as f32).sqrt() as u32;
        if num_neurons != sqrt_num_neurons * sqrt_num_neurons {
            panic!("num_neurons must be a perfect square!");
        }

        let mut neurons = Vec::new();

        for _ in 0..num_neurons {
            neurons.push(Rc::new(Neuron::new(
                fire_threshold,
                max_synapse_weight,
                learning_constant,
            )));
        }

        //Create sensor vec
        let mut sensors = Vec::new();

        for _ in 0..MNIST_AREA {
            sensors.push(Rc::new(NeuronicSensor::new()));
        }

        //Create interneuron synapses
        for i in 0..num_neurons as usize {
            let current_neuron = neurons.get(i).unwrap();

            for j in 0..num_neurons as usize {
                // Make inhibitory synapses with all other neurons
                if i != j {
                    let other_neuron = neurons.get(j).unwrap();

                    current_neuron.create_synapse(
                        neuron_synapse_weight_generator(),
                        SynapticType::Inhibitory,
                        Rc::clone(&other_neuron) as Rc<dyn NeuronicInput>,
                    );
                }
            }
        }

        //Create sensor synapses
        for i in 0..num_neurons as usize {
            let current_neuron = neurons.get(i).unwrap();

            for j in 0..MNIST_AREA {
                let sensor = sensors.get(j).unwrap();

                current_neuron.create_synapse(
                    sensor_synapse_weight_generator(),
                    SynapticType::Excitatory,
                    Rc::clone(&sensor) as Rc<dyn NeuronicInput>,
                );
            }
        }

        Aesir3Layer {
            neurons,
            sensors,
            train_img_vec,
            train_lbl_vec,
        }
    }

    /// This is really a debugging method to ensure this
    /// layer properly loads images into the sensors
    pub fn display_img(&self, img_index: usize) {
        let mut single_img = Vec::new();

        for i in (img_index * MNIST_AREA)..((img_index + 1) * MNIST_AREA) {
            single_img.push((*self.train_img_vec.get(i).unwrap() * 100.) as u32);
        }

        let matrix_img = Matrix::new(MNIST_SIDE_LENGTH, MNIST_SIDE_LENGTH, single_img);

        println!(
            "Img of number {}",
            self.train_lbl_vec.get(img_index).unwrap()
        );
        println!("\n{}", matrix_img);
    }

    fn load_img(&self, img_index: usize) {
        for (i, img_i) in ((img_index * MNIST_AREA)..((img_index + 1) * MNIST_AREA)).enumerate() {
            self.sensors
                .get(i)
                .unwrap()
                .set_measure(*self.train_img_vec.get(img_i).unwrap());
        }
    }

    fn load_external_img(&self, img_index: usize, source: &Vec<f32>) {
        for (i, img_i) in ((img_index * MNIST_AREA)..((img_index + 1) * MNIST_AREA)).enumerate() {
            self.sensors
                .get(i)
                .unwrap()
                .set_measure(*source.get(img_i).unwrap());
        }
    }

    fn train_layer_on_img(&self, img_index: usize, num_cycles: u32) {
        // Load image
        self.load_img(img_index);

        // Clear neurons
        for neuron in self.neurons.iter() {
            neuron.clear();
        }

        // Cycle neurons "num_cycle" times
        let mut cycle = ChargeCycle::Odd;
        for i in 0..num_cycles {
            cycle = cycle.next_cycle();

            let mut values = Vec::new();
            for neuron in self.neurons.iter() {
                values.push(neuron.run_cycle(cycle));
            }

            let side_length = (values.len() as f32).sqrt() as usize;
            let neuron_matrix = Matrix::new(side_length, side_length, values.clone());
            print!("\n{}\n{}", i, neuron_matrix);
        }
        // let mut values = Vec::new();
        //
        // for neuron in self.neurons.iter() {
        //     values.push(neuron.get_measure(cycle));
        // }
        //
        // let side_length = (values.len() as f32).sqrt() as usize;
        // let neuron_matrix = Matrix::new(side_length, side_length, values.clone());
        // print!("\n\n{}", neuron_matrix);
    }

    /// Returns the charge cycle on which it finished
    fn run_static_cycles(&self, num_cycles: u32) -> ChargeCycle {
        // Clear neurons
        for neuron in self.neurons.iter() {
            neuron.clear();
        }

        let mut cycle = ChargeCycle::Even;
        for _ in 0..num_cycles {
            cycle = cycle.next_cycle();

            for neuron in self.neurons.iter() {
                neuron.run_static_cycle(cycle);
            }
        }

        cycle
    }

    /// Does one epoch of training
    pub fn train_layer(&self, num_cycles: u32) {
        for img_index in 0..self.train_lbl_vec.len() {
            self.train_layer_on_img(img_index, num_cycles);

            if img_index % 100 == 0 {
                println!("Finished cycle: {}", img_index);
            }
        }
    }

    pub fn gen_classifier(&self, num_cycles: u32) -> LayerClassifier {
        let mut neuron_class_measures = Vec::new();

        for _ in 0..self.neurons.len() {
            neuron_class_measures.push([0.0; 10]);
        }

        for img_index in 0..self.train_lbl_vec.len() {
            self.load_img(img_index);
            let last_cycle = self.run_static_cycles(num_cycles);

            let label = self.train_lbl_vec.get(img_index).unwrap();
            for (class_measure, neuron) in neuron_class_measures.iter_mut().zip(self.neurons.iter())
            {
                class_measure[*label as usize] += neuron.get_measure(last_cycle);
            }
        }

        let mut neuron_classes = Vec::new();
        let mut class_count = [0_u32; 10];

        for class_measure in neuron_class_measures.iter() {
            let mut max_index = 0;
            let mut max_measure = class_measure[0];

            for i in 1..10 {
                if class_measure[i] > max_measure {
                    max_measure = class_measure[i];
                    max_index = i as u8;
                }
            }

            neuron_classes.push(max_index);
            class_count[max_index as usize] += 1;
        }

        LayerClassifier::new(neuron_classes, class_count)
    }

    pub fn get_classifier_accuracy(
        &self,
        test_images: Vec<f32>,
        test_labels: Vec<u8>,
        classifier: &LayerClassifier,
        num_cycles: u32,
    ) -> f32 {
        let mut num_correct = 0.;

        println!("\nClassifying...\n");

        for img_index in 0..test_labels.len() {
            self.load_external_img(img_index, &test_images);
            let last_cycle = self.run_static_cycles(num_cycles);

            let classification = classifier.classify(&self.neurons, last_cycle);

            if classification == *test_labels.get(img_index).unwrap() {
                num_correct += 1.;
            }

            if img_index % 100 == 0 {
                println!("Finished classification: {}", img_index)
            }
        }

        num_correct / test_labels.len() as f32
    }
}

/// As inspired by the glorious paper, this assigns each neuron
/// in an Aesir3Layer a classifier, which can then be used to classify
/// images.
pub struct LayerClassifier {
    ///Holds the neuron_class of each neuron
    neuron_class: Vec<u8>,

    /// HashMap from digit neuron_class to the
    /// number of neurons in that neuron_class
    class_neuron_count: [u32; 10],
}

impl LayerClassifier {
    pub fn new(neuron_class: Vec<u8>, class_neuron_count: [u32; 10]) -> LayerClassifier {
        LayerClassifier {
            neuron_class,
            class_neuron_count,
        }
    }

    pub fn classify(&self, neuron_layer: &Vec<Rc<Neuron>>, cycle: ChargeCycle) -> u8 {
        // Be sure neuron layer is the same size as the length of this classifier
        if self.neuron_class.len() != neuron_layer.len() {
            panic!("Neuron layer isn't the same size as classifier!");
        }

        let mut total_class_measure = [0.0; 10];

        for (neuron, classification) in neuron_layer.iter().zip(self.neuron_class.iter()) {
            total_class_measure[*classification as usize] += neuron.get_measure(cycle);
        }

        let mut max_index = 0;
        let mut max_average_measure = total_class_measure[0] / self.class_neuron_count[0] as f32;

        for i in 1..10 {
            let average_measure = total_class_measure[i] / self.class_neuron_count[i] as f32;

            if average_measure > max_average_measure {
                max_average_measure = average_measure;
                max_index = i;
            }
        }

        max_index as u8
    }
}

#[cfg(test)]
mod layer_tests;
