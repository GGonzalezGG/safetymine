pub mod yolo;

use anyhow::Result;
use std::sync::Arc;
use yolo::YoloModel;
use common::BoundingBox;
use opencv::prelude::*;
use opencv::{imgproc, videoio};

pub struct VisionEngine {
    model: Arc<YoloModel>,
}

impl VisionEngine {
    pub fn new(model_path: &str) -> Result<Self> {
        let model = Arc::new(YoloModel::new(model_path)?);
        Ok(Self { model })
    }

    pub fn process_frame(&self, frame: &Mat) -> Result<Vec<BoundingBox>> {
        // Convertir Mat (BGR) a DynamicImage (RGB) para YOLO
        let mut rgb_mat = Mat::default();
        imgproc::cvt_color_def(frame, &mut rgb_mat, imgproc::COLOR_BGR2RGB)?;
        
        let size = rgb_mat.size()?;
        let data = rgb_mat.data_bytes()?;
        
        let img = image::ImageBuffer::<image::Rgb<u8>, _>::from_raw(
            size.width as u32,
            size.height as u32,
            data.to_vec()
        ).ok_or_else(|| anyhow::anyhow!("Failed to create image buffer"))?;
        
        let dyn_img = image::DynamicImage::ImageRgb8(img);
        self.model.predict(&dyn_img)
    }
}

pub struct SafetyGuiApp {
    engine: Arc<VisionEngine>,
    video_capture: Option<videoio::VideoCapture>,
    static_image_processed: bool,
    last_detections: Vec<BoundingBox>,
    current_frame: Option<egui::ColorImage>,
}

impl SafetyGuiApp {
    pub fn new(_cc: &eframe::CreationContext<'_>, model_path: &str, media_path: &str) -> Self {
        let engine = Arc::new(VisionEngine::new(model_path).expect("Failed to load model"));
        
        // Detectamos si es la cámara (índice 0) o un archivo de video
        let is_camera = media_path == "0";
        let is_video = is_camera || media_path.ends_with(".mp4") || media_path.ends_with(".avi");
        
        let mut video_capture = None;
        let mut current_frame = None;
        let mut last_detections = Vec::new();
        let mut static_image_processed = false;
        
        if is_video {
            if is_camera {
                // Abrir la cámara web de hardware (pasando 0 como i32)
                video_capture = Some(videoio::VideoCapture::new(0, videoio::CAP_ANY).expect("Failed to open webcam"));
            } else {
                // Abrir archivo de video local
                video_capture = Some(videoio::VideoCapture::from_file(media_path, videoio::CAP_ANY).expect("Failed to open video file"));
            }
        } else {
            // Es una imagen (ej: .webp, .jpg, .png)
            if let Ok(dyn_img) = image::open(media_path) {
                if let Ok(detections) = engine.model.predict(&dyn_img) {
                    last_detections = detections;
                    
                    let rgb_img = dyn_img.to_rgb8();
                    let size = [rgb_img.width() as _, rgb_img.height() as _];
                    let pixels = rgb_img.into_flat_samples();
                    current_frame = Some(egui::ColorImage::from_rgb(size, pixels.as_slice()));
                    static_image_processed = true;
                }
            } else {
                println!("Error al cargar la imagen: {}", media_path);
            }
        }
        
        Self {
            engine,
            video_capture,
            static_image_processed,
            last_detections,
            current_frame,
        }
    }
}

impl eframe::App for SafetyGuiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Leer siguiente frame SOLO si es un video
        if let Some(cap) = &mut self.video_capture {
            let mut mat = Mat::default();
            if cap.read(&mut mat).unwrap_or(false) && !mat.empty() {
                if let Ok(detections) = self.engine.process_frame(&mat) {
                    self.last_detections = detections;
                }
                
                // Convertir para visualización
                let mut rgb_mat = Mat::default();
                let _ = imgproc::cvt_color_def(&mat, &mut rgb_mat, imgproc::COLOR_BGR2RGB);
                if let Ok(size) = rgb_mat.size() {
                    if let Ok(bytes) = rgb_mat.data_bytes() {
                        self.current_frame = Some(egui::ColorImage::from_rgb(
                            [size.width as usize, size.height as usize],
                            bytes
                        ));
                    }
                }
            }
            ctx.request_repaint(); // Mantener el video fluyendo
        } else if !self.static_image_processed {
            // Si es imagen pero no se procesó, no pedimos repaint continuo
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("SafetyMine Checkpoint - Visión Inteligente");
            
            if let Some(img) = &self.current_frame {
                let texture = ui.ctx().load_texture("frame", img.clone(), Default::default());
                
                // Usamos un Frame o simplemente calculamos la posición
                let (rect, _response) = ui.allocate_at_least(ui.available_size(), egui::Sense::hover());
                
                // Dibujar la imagen
                ui.painter().image(texture.id(), rect, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), egui::Color32::WHITE);

                // Dibujar las cajas
                let painter = ui.painter_at(rect);
                let img_w = img.width() as f32;
                let img_h = img.height() as f32;
                
                for det in &self.last_detections {
                    // Escalar coordenadas de la imagen original al tamaño del widget en pantalla
                    let x1 = (det.x1 / img_w) * rect.width() + rect.left();
                    let y1 = (det.y1 / img_h) * rect.height() + rect.top();
                    let x2 = (det.x2 / img_w) * rect.width() + rect.left();
                    let y2 = (det.y2 / img_h) * rect.height() + rect.top();
                    
                    let box_rect = egui::Rect::from_min_max(egui::pos2(x1, y1), egui::pos2(x2, y2));
                    
                    // Color según etiqueta
                    let color = match det.label {
                        common::EppItem::Casco | common::EppItem::Mascarilla | common::EppItem::ChalecoSeguridad => egui::Color32::GREEN,
                        common::EppItem::SinCasco | common::EppItem::SinMascarilla | common::EppItem::SinChaleco => egui::Color32::RED,
                        common::EppItem::Persona => egui::Color32::BLUE,
                        _ => egui::Color32::YELLOW,
                    };
                    
                    painter.rect_stroke(box_rect, 2.0, egui::Stroke::new(3.0, color));
                    painter.text(
                        box_rect.left_top(),
                        egui::Align2::LEFT_TOP,
                        format!("{:?} ({:.2})", det.label, det.score),
                        egui::FontId::proportional(16.0),
                        egui::Color32::WHITE,
                    );
                }
            }

            ui.separator();
            ui.label(format!("Detecciones activas: {}", self.last_detections.len()));
        });
    }
}
