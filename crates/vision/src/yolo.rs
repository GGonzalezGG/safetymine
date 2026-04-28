use anyhow::Result;
use common::{BoundingBox, EppItem};
use ndarray::Array;
use ort::{session::Session, value::Value};
use image::{imageops::FilterType, DynamicImage, GenericImageView};

use std::sync::Mutex;

pub struct YoloModel {
    session: Mutex<Session>,
}

impl YoloModel {
    pub fn new(model_path: &str) -> Result<Self> {
        let session = Session::builder()?
            .commit_from_file(model_path)?;
        
        Ok(Self { session: Mutex::new(session) })
    }

    pub fn predict(&self, img: &DynamicImage) -> Result<Vec<BoundingBox>> {
        let (width, height) = img.dimensions();
        
        // 1. Pre-procesamiento: Resize a 640x640 y normalizar
        let resized = img.resize_exact(640, 640, FilterType::Triangle);
        let mut input = Array::zeros((1, 3, 640, 640));
        for (x, y, pixel) in resized.pixels() {
            input[[0, 0, y as usize, x as usize]] = (pixel.0[0] as f32) / 255.0;
            input[[0, 1, y as usize, x as usize]] = (pixel.0[1] as f32) / 255.0;
            input[[0, 2, y as usize, x as usize]] = (pixel.0[2] as f32) / 255.0;
        }

        // 2. Inferencia
        let input_tensor = Value::from_array((
            vec![1, 3, 640, 640],
            input.into_raw_vec()
        ))?;
        let mut session_lock = self.session.lock().unwrap();
        let outputs = session_lock.run(ort::inputs![input_tensor])?;
        
        // ort 2.0 devuelve (Shape, &[T]) al extraer el tensor
        let (shape_info, data) = outputs[0].try_extract_tensor::<f32>()?;
        
        let num_classes = 10; 
        let mut detections = Vec::new();
        let num_boxes = shape_info[2] as usize;

        // El mapeo de clases se maneja ahora en la función match_class

        // 3. Post-procesamiento (Acceso directo a memoria plana, C-order: [batch, channels, items])
        for i in 0..num_boxes {
            let cx = data[0 * num_boxes + i];
            let cy = data[1 * num_boxes + i];
            let w  = data[2 * num_boxes + i];
            let h  = data[3 * num_boxes + i];

            // Encontrar la clase con mayor score
            let mut max_score = 0.0;
            let mut class_id = 0;
            for j in 0..num_classes {
                let score = data[(4 + j) * num_boxes + i];
                if score > max_score {
                    max_score = score;
                    class_id = j;
                }
            }

            if max_score > 0.5 {
                let x1 = (cx - w / 2.0) * (width as f32 / 640.0);
                let y1 = (cy - h / 2.0) * (height as f32 / 640.0);
                let x2 = (cx + w / 2.0) * (width as f32 / 640.0);
                let y2 = (cy + h / 2.0) * (height as f32 / 640.0);

                detections.push(BoundingBox {
                    x1,
                    y1,
                    x2,
                    y2,
                    score: max_score,
                    label: match_class(class_id),
                });
            }
        }

        // 4. Aplicar NMS (Non-Maximum Suppression)
        Ok(self.nms(detections, 0.45))
    }

    fn nms(&self, mut detections: Vec<BoundingBox>, iou_threshold: f32) -> Vec<BoundingBox> {
        detections.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        let mut keep = Vec::new();

        while !detections.is_empty() {
            let best = detections.remove(0);
            keep.push(best.clone());
            detections.retain(|item| {
                intersection_over_union(&best, item) < iou_threshold
            });
        }
        keep
    }
}

fn intersection_over_union(a: &BoundingBox, b: &BoundingBox) -> f32 {
    let x1 = a.x1.max(b.x1);
    let y1 = a.y1.max(b.y1);
    let x2 = a.x2.min(b.x2);
    let y2 = a.y2.min(b.y2);

    let intersection_area = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    let area_b = (b.x2 - b.x1) * (b.y2 - b.y1);

    intersection_area / (area_a + area_b - intersection_area)
}

fn match_class(id: usize) -> EppItem {
    match id {
        0 => EppItem::Casco,
        1 => EppItem::Mascarilla,
        2 => EppItem::SinCasco,
        3 => EppItem::SinMascarilla,
        4 => EppItem::SinChaleco,
        5 => EppItem::Persona,
        6 => EppItem::ConoSeguridad,
        7 => EppItem::ChalecoSeguridad,
        8 => EppItem::Maquinaria,
        9 => EppItem::Vehiculo,
        _ => EppItem::Persona, // Fallback
    }
}
