use slint::{Model, VecModel, SharedString, Image, Rgba8Pixel, SharedPixelBuffer};
use std::sync::{Arc, Mutex};
use std::rc::Rc;
use std::thread;
use std::time::Duration; // <-- Corregido
use image::DynamicImage; // <-- Corregido

use vision::{VisionEngine, CameraStream};
use safety_engine::SafetyEvaluator;
use common::MinaContext;

slint::include_modules!();

fn main() -> Result<(), slint::PlatformError> {
    println!("Iniciando SafetyMine GUI...");
    
    // 1. Cargar la DB inicial para poblar la UI
    let db_path = "data/db.json"; 
    let db_content = std::fs::read_to_string(db_path).expect("No se pudo cargar db.json");
    let db: serde_json::Value = serde_json::from_str(&db_content).expect("Error en JSON");
    
    // Extraer nombres de minas para el ComboBox
    let nombres: Vec<SharedString> = db["minas"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["nombre_mina"].as_str().unwrap().into())
        .collect();

    // 2. Estado compartido del evaluador
    let primer_id = nombres[0].to_string();
    let evaluator = Arc::new(Mutex::new(SafetyEvaluator::new(db_path, &primer_id)));

    // 3. Inicializamos la UI (¡SOLO UNA VEZ!)
    let ui = SafetyWindow::new()?;
    ui.set_lista_minas(Rc::new(VecModel::from(nombres)).into());

    // --- LÓGICA: Selección de Mina ---
    let ui_handle_mina = ui.as_weak();
    let evaluator_mina = evaluator.clone();
    
    ui.on_mina_seleccionada(move |nombre| {
        let mut ev = evaluator_mina.lock().unwrap();
        *ev = SafetyEvaluator::new("data/db.json", nombre.as_str());
        
        if let Some(ui) = ui_handle_mina.upgrade() {
            if let Some(ctx) = &ev.contexto_actual {
                // CORRECCIÓN: riesgos_activos con "e"
                let riesgos: Vec<SharedString> = ctx.riesgos_activos.iter().map(|r| r.as_str().into()).collect();
                ui.set_riesgos_mina(Rc::new(VecModel::from(riesgos)).into());

                // Actualizar Checklist EPP 
                let epp: Vec<SharedString> = ctx.epp_obligatorio.iter().map(|e| format!("{:?}", e).into()).collect();
                ui.set_lista_epp(Rc::new(VecModel::from(epp)).into());
            }
        }
    });

    // 4. Inicializamos YOLO 
    println!("[Sistema] Cargando modelo ONNX...");
    let engine = Arc::new(VisionEngine::new("best.onnx").expect("Fallo al cargar el modelo YOLO"));
    
    // Creamos el buzón para guardar el frame de la cámara
    let ultimo_frame = Arc::new(Mutex::new(None::<DynamicImage>));

    // --- LÓGICA: Cambiar Fuente (RFD) ---
    let ui_handle_fuente = ui.as_weak();
    
    ui.on_cambiar_fuente(move || {
        let ui_weak = ui_handle_fuente.clone();
        
        // Lanzamos el explorador en un hilo secundario para NO congelar la interfaz
        thread::spawn(move || {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("Multimedia", &["mp4", "avi", "jpg", "png", "webp"])
                .set_title("Seleccionar fuente de prueba")
                .pick_file() 
            {
                let nombre_archivo = path.file_name().unwrap().to_string_lossy().to_string();
                let ruta_completa = path.display().to_string();
                
                // Volvemos al hilo principal solo para actualizar el texto en pantalla
                let _ = slint::invoke_from_event_loop(move || {
                    if let Some(ui) = ui_weak.upgrade() {
                        ui.set_fuente_actual(nombre_archivo.into());
                        
                        // [!] IMPRIME LA RUTA EN CONSOLA PARA LA FASE 3
                        println!("[UI] Archivo seleccionado listo para análisis: {}", ruta_completa);
                    }
                });
            }
        });
    });

    // Forzar actualización inicial de la base de datos visual
    ui.invoke_mina_seleccionada(db["minas"][0]["nombre_mina"].as_str().unwrap().into());

    // --- HILO DE VIDEO EN VIVO ---
    let ui_handle_video = ui.as_weak();
    let ultimo_frame_bg = ultimo_frame.clone();

    thread::spawn(move || {
        println!("[Sistema] Abriendo cámara nativa en hilo de fondo...");
        let mut cam = CameraStream::new(0).expect("Fallo al abrir la cámara web");

        loop {
            if let Ok(img) = cam.get_frame() {
                if let Ok(mut buz) = ultimo_frame_bg.lock() {
                    *buz = Some(img.clone());
                }

                let rgba = img.into_rgba8();
                let buffer = SharedPixelBuffer::<Rgba8Pixel>::clone_from_slice(
                    rgba.as_raw(),
                    rgba.width(),
                    rgba.height(),
                );

                let _ = slint::invoke_from_event_loop({
                    let ui_weak = ui_handle_video.clone();
                    move || {
                        if let Some(ui) = ui_weak.upgrade() {
                            let slint_img = Image::from_rgba8(buffer);
                            ui.set_frame_camara(slint_img);
                            ui.set_fuente_actual("Cámara Web (En Vivo)".into());
                        }
                    }
                });
            }
            thread::sleep(Duration::from_millis(33)); 
        }
    });

    // --- BOTÓN VALIDAR ---
    let engine_clone = engine.clone();
    let ultimo_frame_btn = ultimo_frame.clone();

    ui.on_validar_manual(move || {
        println!("\n--- INICIANDO PRUEBA DE VISIÓN ---");
        let frame_actual = ultimo_frame_btn.lock().unwrap().clone();

        match frame_actual {
            Some(img) => {
                println!("[1/2] Fotograma leído del buffer.");
                match engine_clone.process_frame(&img) {
                    Ok(detections) => {
                        println!("[2/2] Inferencia completada. Detecciones: {}", detections.len());
                        for (i, det) in detections.iter().enumerate() {
                            println!("  -> #{}: {:?} (Confianza: {:.1}%)", 
                                i + 1, det.label, det.score * 100.0);
                        }
                    }
                    Err(e) => println!("Error procesando frame en YOLO: {}", e),
                }
            }
            None => println!("Aún no hay fotogramas de la cámara."),
        }
        println!("----------------------------------\n");
    });

    ui.run()
}