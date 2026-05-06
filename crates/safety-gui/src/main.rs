use slint::{ VecModel, SharedString, Image, Rgba8Pixel, SharedPixelBuffer};
use std::sync::{Arc, Mutex};
use std::rc::Rc;
use std::thread;
use std::time::{Duration}; 
use image::DynamicImage;
use image::Rgba; 

// Importaciones de dibujo
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::rect::Rect;

use vision::{VisionEngine, CameraStream};
use safety_engine::SafetyEvaluator;

// IA Local
use std::sync::atomic::{AtomicBool, Ordering};
use ai_brain::TutorBrain;

slint::include_modules!();

fn main() -> Result<(), slint::PlatformError> {
    println!("Iniciando SafetyMine GUI...");
    
    let db_path = "data/db.json"; 
    let db_content = std::fs::read_to_string(db_path).expect("No se pudo cargar db.json");
    let db: serde_json::Value = serde_json::from_str(&db_content).expect("Error en JSON");
    
    let nombres: Vec<SharedString> = db["minas"].as_array().unwrap().iter()
        .map(|m| m["nombre_mina"].as_str().unwrap().into()).collect();

    let primer_id = nombres[0].to_string();
    let evaluator = Arc::new(Mutex::new(SafetyEvaluator::new(db_path, &primer_id)));

    let ui = SafetyWindow::new()?;
    ui.set_lista_minas(Rc::new(VecModel::from(nombres)).into());

    let ui_handle_mina = ui.as_weak();
    let evaluator_mina = evaluator.clone();
    
    ui.on_mina_seleccionada(move |nombre| {
        let mut ev = evaluator_mina.lock().unwrap();
        *ev = SafetyEvaluator::new("data/db.json", nombre.as_str());
        
        if let Some(ui) = ui_handle_mina.upgrade() {
            if let Some(ctx) = &ev.contexto_actual {
                let riesgos: Vec<SharedString> = ctx.riesgos_activos.iter().map(|r| r.as_str().into()).collect();
                ui.set_riesgos_mina(Rc::new(VecModel::from(riesgos)).into());

                let epp: Vec<SharedString> = ctx.epp_obligatorio.iter().map(|e| format!("{:?}", e).into()).collect();
                ui.set_lista_epp(Rc::new(VecModel::from(epp)).into());
            }
        }
    });

    //Modelo ONNX de Yolo para detectar el equipamiento de seguridad
    println!("[Sistema] Cargando modelo ONNX...");
    let engine = Arc::new(VisionEngine::new("best.onnx").expect("Fallo al cargar YOLO"));
    
    // --- LLM Local ---
    println!("[Sistema] Despertando al Safety Buddy (LLM)...");
    
    let brain = Arc::new(Mutex::new(TutorBrain::new("assets/models/gemma-4-E2B-it-Q4_K_M.gguf").expect("Fallo al cargar LLM")));
    
    // Memoria para no repetirle lo mismo al LLM
    let ultimos_faltantes = Arc::new(Mutex::new(None::<Vec<common::EppItem>>));
    
    // Seguro para no lanzar 2 inferencias al mismo tiempo y quemar la CPU
    let llm_ocupado = Arc::new(AtomicBool::new(false)); 

    let historial_mensajes = Arc::new(Mutex::new(Vec::<slint::SharedString>::new()));
    // ---------------------------

    let ultimo_frame = Arc::new(Mutex::new(None::<DynamicImage>));
    let modo_fuente = Arc::new(Mutex::new(String::new())); 

    // --- LÓGICA: Volver a la Cámara ---
    let modo_fuente_cam_btn = modo_fuente.clone();
    ui.on_usar_camara(move || {
        println!("[UI] Volviendo a la cámara en vivo...");
        *modo_fuente_cam_btn.lock().unwrap() = String::new(); 
    });

    // --- LÓGICA: Cambiar Fuente (Archivos) ---
    let ui_handle_fuente = ui.as_weak();
    let ultimo_frame_rfd = ultimo_frame.clone();
    let modo_fuente_rfd = modo_fuente.clone();
    
    ui.on_cambiar_fuente(move || {
        let ui_weak = ui_handle_fuente.clone();
        let ultimo_frame_bg = ultimo_frame_rfd.clone();
        let modo_bg = modo_fuente_rfd.clone();
        
        thread::spawn(move || {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("Imágenes", &["jpg", "jpeg", "png", "webp"])
                .set_title("Seleccionar imagen de prueba")
                .pick_file() 
            {
                let ruta_completa = path.display().to_string();
                let nombre_archivo = path.file_name().unwrap().to_string_lossy().to_string();
                
                *modo_bg.lock().unwrap() = ruta_completa.clone();

                if let Ok(img) = image::open(&path) {
                    *ultimo_frame_bg.lock().unwrap() = Some(img.clone());
                    let rgba = img.into_rgba8();
                    let buffer = SharedPixelBuffer::<Rgba8Pixel>::clone_from_slice(
                        rgba.as_raw(), rgba.width(), rgba.height()
                    );
                    let _ = slint::invoke_from_event_loop(move || {
                        if let Some(ui) = ui_weak.upgrade() {
                            ui.set_frame_camara(Image::from_rgba8(buffer));
                            ui.set_fuente_actual(format!("Archivo: {}", nombre_archivo).into());
                            ui.invoke_validar_manual();
                        }
                    });
                }
            }
        });
    });

    ui.invoke_mina_seleccionada(db["minas"][0]["nombre_mina"].as_str().unwrap().into());

    // ==============================================================
    // --- ESTADO COMPARTIDO PARA LA TERCERA VÍA (IA VS CÁMARA) ---
    // ==============================================================
    // Aquí guardaremos: (Las cajas detectadas, Si el sistema es seguro, El mensaje del tutor)
    // Usamos Vec::<common::BoundingBox>::new() para decirle el tipo exacto
    let estado_inferencia = Arc::new(Mutex::new((Vec::<common::BoundingBox>::new(), true, String::new())));

    // ==============================================================
    // --- HILO 1: CÁMARA EXPRESS (Dibuja a 30 FPS sin bloqueos) ---
    // ==============================================================
    let ui_handle_video = ui.as_weak();
    let frame_para_inferencia = ultimo_frame.clone();
    let modo_fuente_cam = modo_fuente.clone();
    let estado_dibujo = estado_inferencia.clone();

    thread::spawn(move || {
        if let Ok(mut cam) = CameraStream::new(0) {
            loop {
                let es_camara = modo_fuente_cam.lock().unwrap().is_empty();
                
                if es_camara {
                    if let Ok(img) = cam.get_frame() {
                        // 1. Dejamos una copia fresquita en el buzón para que YOLO la tome cuando quiera
                        if let Ok(mut buz) = frame_para_inferencia.lock() { 
                            *buz = Some(img.clone()); 
                        }

                        let mut img_rgba = img.to_rgba8();

                        // 2. Leemos la última orden de YOLO (Instantáneo, no bloquea la cámara)
                        let (ultimas_detecciones, sistema_seguro, _) = estado_dibujo.lock().unwrap().clone();

                        // 3. DIBUJAMOS LAS CAJAS GRUESAS
                        for det in &ultimas_detecciones {
                            let color = if sistema_seguro { 
                                Rgba([0, 255, 0, 255]) 
                            } else { 
                                Rgba([255, 0, 0, 255]) 
                            };
                            
                            // TRUCO DE GROSOR: Dibujamos 4 rectángulos concéntricos
                            for grosor in 0..4 {
                                let rect = Rect::at(det.x1 as i32 - grosor, det.y1 as i32 - grosor)
                                    .of_size(
                                        (det.x2 - det.x1) as u32 + (grosor as u32 * 2), 
                                        (det.y2 - det.y1) as u32 + (grosor as u32 * 2)
                                    );
                                draw_hollow_rect_mut(&mut img_rgba, rect, color);
                            }
                        }

                        // 4. Enviamos a la interfaz visual
                        let buffer = SharedPixelBuffer::<Rgba8Pixel>::clone_from_slice(
                            img_rgba.as_raw(), img_rgba.width(), img_rgba.height()
                        );

                        let _ = slint::invoke_from_event_loop({
                            let ui_weak = ui_handle_video.clone();
                            move || {
                                if let Some(ui) = ui_weak.upgrade() {
                                    ui.set_frame_camara(Image::from_rgba8(buffer));
                                    ui.set_fuente_actual("Cámara Web (Auto-Validación)".into());
                                }
                            }
                        });
                    }
                }
                // Respiro vital de 16ms para no quemar la CPU si la cámara está apagada (modo archivo)
                thread::sleep(Duration::from_millis(16)); 
            }
        }
    });

    // ==============================================================
    // --- HILO 2: EL CEREBRO IA (YOLO + LLM) ---
    // ==============================================================
    let engine_bg = engine.clone();
    let evaluator_bg = evaluator.clone();
    let frame_de_camara = ultimo_frame.clone();
    let estado_yolo = estado_inferencia.clone();
    let ui_handle_tutor = ui.as_weak();
    
    // Clones para el LLM
    let brain_bg = brain.clone();
    let memoria_faltantes = ultimos_faltantes.clone();
    let flag_ocupado = llm_ocupado.clone();

    thread::spawn(move || {
        loop {
            let frame_actual = frame_de_camara.lock().unwrap().clone();
            
            if let Some(img) = frame_actual {
                if let Ok(detecciones) = engine_bg.process_frame(&img) {
                    
                    let ev_lock = evaluator_bg.lock().unwrap();
                    let faltantes = ev_lock.verificar_faltantes(&common::DetectionResult { 
                        detecciones: detecciones.clone(),
                        timestamp: 0 
                    });

                    let sistema_seguro = faltantes.is_empty();

                    // Guardamos resultados de visión para que la cámara dibuje cajas
                    if let Ok(mut estado) = estado_yolo.lock() {
                        *estado = (detecciones, sistema_seguro, String::new());
                    }

                    // --- LÓGICA DE GATILLO LLM (DEBOUNCE) ---
                    let mut ha_cambiado = false;
                    if let Ok(mut ultimos) = memoria_faltantes.lock() {
                        // Solo disparamos si la lista de EPP faltantes es diferente a la última vez
                        if ultimos.as_ref() != Some(&faltantes) {
                            ha_cambiado = true;
                            *ultimos = Some(faltantes.clone()); // Actualizamos la memoria
                        }
                    }

                    // Si hubo un cambio y el LLM no está ocupado pensando otra respuesta...
                    if ha_cambiado && !flag_ocupado.load(Ordering::SeqCst) {
                        flag_ocupado.store(true, Ordering::SeqCst); // Bloqueamos la puerta

                        // Extraemos los datos necesarios para el prompt
                        let contexto_actual = ev_lock.contexto_actual.clone().unwrap();
                        let protocolo_texto = ev_lock.obtener_texto_protocolo();
                        let faltantes_llm = faltantes.clone();
                        
                        let brain_llm = brain_bg.clone();
                        let flag_llm = flag_ocupado.clone();
                        let ui_llm = ui_handle_tutor.clone();

                        // Lanzamos al LLM en un sub-hilo para que YOLO siga analizando cajas a 1 FPS
                        thread::spawn(move || {
                            // 1. Avisamos en pantalla que la IA está escribiendo
                            let _ = slint::invoke_from_event_loop({
                                let ui_weak = ui_llm.clone();
                                move || {
                                    if let Some(ui) = ui_weak.upgrade() {
                                        ui.set_historial_tutor(Rc::new(VecModel::from(vec!["🤖 Safety Buddy está analizando la situación...".into()])).into());
                                    }
                                }
                            });

                            // 2. Ejecutamos Llama 3
                            if let Ok(brain_lock) = brain_llm.lock() {
                                match brain_lock.evaluate_safety(&contexto_actual, &faltantes_llm, &protocolo_texto) {
                                    Ok(respuesta_ia) => {
                                        // 3. Enviamos la respuesta pedagógica a la Interfaz de Slint
                                        let prefijo = if respuesta_ia.is_safe { "✅" } else { "❌" };
                                        let mensaje_final = format!("{} [{}]\n{}", prefijo, respuesta_ia.severity, respuesta_ia.explanation);

                                        let _ = slint::invoke_from_event_loop({
                                            let ui_weak = ui_llm.clone();
                                            move || {
                                                if let Some(ui) = ui_weak.upgrade() {
                                                    ui.set_historial_tutor(Rc::new(VecModel::from(vec![mensaje_final.into()])).into());
                                                }
                                            }
                                        });
                                    }
                                    Err(e) => println!("[Error LLM] {}", e),
                                }
                            }
                            // Liberamos la puerta para futuras inferencias
                            flag_llm.store(false, Ordering::SeqCst); 
                        });
                    }
                }
            }
            thread::sleep(Duration::from_millis(1000));
        }
    });

    // --- BOTÓN VALIDAR MANUAL (Dibuja sobre archivos estáticos) ---
    let engine_clone = engine.clone();
    let ultimo_frame_btn = ultimo_frame.clone();
    let evaluator_btn = evaluator.clone();
    let ui_handle_validar = ui.as_weak(); // Para poder actualizar la UI desde aquí

    let brain_btn = brain.clone();

    ui.on_validar_manual(move || {
        println!("\n--- INICIANDO PRUEBA MANUAL / IMAGEN ESTÁTICA ---");
        let frame_actual = ultimo_frame_btn.lock().unwrap().clone();
        
        match frame_actual {
            Some(img) => {
                match engine_clone.process_frame(&img) {
                    Ok(detecciones) => {
                        let ev = evaluator_btn.lock().unwrap();
                        let faltantes = ev.verificar_faltantes(&common::DetectionResult { 
                            detecciones: detecciones.clone(), 
                            timestamp: 0 
                        });
                        
                        let sistema_seguro = faltantes.is_empty();
                        
                        if sistema_seguro {
                            println!("✅ Cumplimiento total.");
                        } else {
                            println!("❌ PELIGRO: Faltan EPP: {:?}", faltantes);
                        }

                        let contexto_actual = ev.contexto_actual.clone().unwrap();
                        let protocolo_texto = ev.obtener_texto_protocolo();

                        // 1. DIBUJAMOS EN LA IMAGEN ESTÁTICA
                        let mut img_rgba = img.to_rgba8();
                        for det in &detecciones {
                            let color = if sistema_seguro { 
                                Rgba([0, 255, 0, 255]) 
                            } else { 
                                Rgba([255, 0, 0, 255]) 
                            };
                            
                            for grosor in 0..4 {
                                let rect = Rect::at(det.x1 as i32 - grosor, det.y1 as i32 - grosor)
                                    .of_size(
                                        (det.x2 - det.x1) as u32 + (grosor as u32 * 2), 
                                        (det.y2 - det.y1) as u32 + (grosor as u32 * 2)
                                    );
                                draw_hollow_rect_mut(&mut img_rgba, rect, color);
                            }
                        }

                        // 2. ACTUALIZAMOS LA UI (Imagen y Tutor)
                        if let Some(ui) = ui_handle_validar.upgrade() {
                            let buffer = SharedPixelBuffer::<Rgba8Pixel>::clone_from_slice(
                                img_rgba.as_raw(), img_rgba.width(), img_rgba.height()
                            );
                            ui.set_frame_camara(Image::from_rgba8(buffer));
                        }

                        let brain_llm = brain_btn.clone();
                        let ui_llm = ui_handle_validar.clone();
                        let faltantes_llm = faltantes.clone();

                        thread::spawn(move || {
                            // A) Mostramos mensaje temporal de carga
                            let _ = slint::invoke_from_event_loop({
                                let ui_weak = ui_llm.clone();
                                move || { 
                                    if let Some(ui) = ui_weak.upgrade() { 
                                        ui.set_historial_tutor(Rc::new(VecModel::from(vec!["🤖 Safety Buddy está analizando la imagen estática...".into()])).into()); 
                                    } 
                                }
                            });

                            // B) Ejecutamos Llama 3
                            if let Ok(brain_lock) = brain_llm.lock() {
                                if let Ok(respuesta_ia) = brain_lock.evaluate_safety(&contexto_actual, &faltantes_llm, &protocolo_texto) {
                                    let prefijo = if respuesta_ia.is_safe { "✅" } else { "❌" };
                                    let mensaje_final = format!("{} [{}]\n{}", prefijo, respuesta_ia.severity, respuesta_ia.explanation);

                                    // C) Imprimimos la respuesta final en la consola de Slint
                                    let _ = slint::invoke_from_event_loop({
                                        let ui_weak = ui_llm.clone();
                                        move || { 
                                            if let Some(ui) = ui_weak.upgrade() { 
                                                ui.set_historial_tutor(Rc::new(VecModel::from(vec![mensaje_final.into()])).into()); 
                                            } 
                                        }
                                    });
                                }
                            }
                        });
                    }
                    Err(e) => println!("Error YOLO: {}", e),
                }
            }
            None => println!("Aún no hay imagen."),
        }
        println!("-------------------------------------------------\n");
    });

    ui.run()
}