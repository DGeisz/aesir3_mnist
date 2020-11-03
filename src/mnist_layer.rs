use aesir3::neuron::{Neuronic, SynapticType, NeuronicInput};
use aesir3::{Neuron, NeuronicSensor};
use std::rc::Rc;

const MNIST_SIDE_LENGTH: u32 = 28;
const MNIST_AREA: u32 = MNIST_SIDE_LENGTH * MNIST_SIDE_LENGTH;

pub struct Aesir3Layer {
    neurons: Vec<Rc<Neuron>>,
    sensors: Vec<Rc<NeuronicSensor>>,
    train_img_vec: Vec<u8>,
    train_lbl_vec: Vec<u8>,
}

impl Aesir3Layer {
    pub fn new(
        train_img_vec: Vec<u8>, // This should already be modified so values are between 0 and 1
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

            for j in 0..MNIST_AREA as usize {
                let sensor = sensors.get(j).unwrap();

                current_neuron.create_synapse(
                    sensor_synapse_weight_generator(),
                    SynapticType::Excitatory,
                    Rc::clone(&sensor) as Rc<dyn NeuronicInput>
                );
            }
        }

        unimplemented!();
    }
}
