use eframe;
use anyhow::Result;

fn main() -> Result<(), eframe::Error> {
    // Configuración de la ventana egui
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([800.0, 600.0])
            .with_title("SafetyMine - Monitoreo en Tiempo Real"),
        ..Default::default()
    };

    // RUTAS A ARCHIVOS:
    let model_path = "best.onnx"; 
    
    // NOTA: Puedes usar un video (.mp4) o una imagen estática (.webp, .jpg, .png).
    // Solo cambia esta ruta por la de tu archivo.
    let media_path = "data/fototest.webp"; 

    println!("Iniciando SafetyMine Checkpoint...");
    println!("Cargando modelo YOLOv8 desde: {}", model_path);
    println!("Cargando archivo multimedia desde: {}", media_path);

    eframe::run_native(
        "SafetyMine Checkpoint",
        options,
        Box::new(move |cc| {
            // Inicializar el motor de visión (este llamará internamente a ort y opencv)
            Box::new(vision::SafetyGuiApp::new(cc, model_path, media_path))
        }),
    )
}
