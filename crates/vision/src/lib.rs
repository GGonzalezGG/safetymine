pub mod yolo;

use anyhow::Result;
use std::sync::Arc;
use image::DynamicImage;
use common::BoundingBox;
use yolo::YoloModel;

// Importaciones de Nokhwa para la cámara nativa
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};
use nokhwa::Camera;

pub struct VisionEngine {
    model: Arc<YoloModel>,
}

impl VisionEngine {
    pub fn new(model_path: &str) -> Result<Self> {
        let model = Arc::new(YoloModel::new(model_path)?);
        Ok(Self { model })
    }

    // ¡Mira qué limpio! Ya no hay Mat ni conversiones BGR a RGB raras.
    // Recibe la imagen de Rust, y se la pasa a YOLO.
    pub fn process_frame(&self, img: &DynamicImage) -> Result<Vec<BoundingBox>> {
        self.model.predict(img)
    }
}

// Un nuevo componente para manejar la cámara sin OpenCV
pub struct CameraStream {
    camera: Camera,
}

impl CameraStream {
    pub fn new(index: u32) -> Result<Self> {
        // Pedimos la cámara con la mayor tasa de cuadros posible en RGB
        let format = RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
        let mut camera = Camera::new(CameraIndex::Index(index), format)
            .map_err(|e| anyhow::anyhow!("Error al inicializar cámara: {}", e))?;
        
        camera.open_stream()
            .map_err(|e| anyhow::anyhow!("Error al abrir stream de cámara: {}", e))?;

        Ok(Self { camera })
    }

    // Lee un frame y lo entrega listo para procesar en YOLO y dibujar en Slint
    pub fn get_frame(&mut self) -> Result<DynamicImage> {
        let frame = self.camera.frame()
            .map_err(|e| anyhow::anyhow!("Error leyendo fotograma: {}", e))?;
        
        let decoded = frame.decode_image::<RgbFormat>()
            .map_err(|e| anyhow::anyhow!("Error decodificando fotograma: {}", e))?;
            
        Ok(DynamicImage::ImageRgb8(decoded))
    }
}