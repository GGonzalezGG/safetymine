use common::{MinaContext, EppItem, AIResponse};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
#[allow(deprecated)]
use llama_cpp_2::model::Special;
use llama_cpp_2::sampling::LlamaSampler;

use std::num::NonZeroU32;

pub mod prompts;

pub struct TutorBrain {
    backend: LlamaBackend,
    model: LlamaModel,
}

impl TutorBrain {
    pub fn new(model_path: &str) -> Result<Self, String> {
        let backend = LlamaBackend::init()
            .map_err(|e| format!("Error iniciando backend Llama: {}", e))?;
        
        let model_params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
            .map_err(|e| format!("Error cargando modelo en {}: {}", model_path, e))?;

        Ok(Self { backend, model })
    }

    pub fn evaluate_safety(
        &self, 
        contexto: &MinaContext, 
        faltantes: &[EppItem],
        protocolo_texto: &str
    ) -> Result<AIResponse, String> {
        
        let system_prompt = prompts::build_system_prompt();
        let user_prompt = prompts::build_user_prompt(contexto, faltantes, protocolo_texto);
        
        // Formato estándar (Llama 3)
        let full_prompt = format!(
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{}\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            system_prompt, user_prompt
        );

        let response_text = self.run_inference(&full_prompt)?;
        self.parse_json_response(&response_text)
    }

    fn run_inference(&self, prompt: &str) -> Result<String, String> {
        // AMPLIAMOS LA MEMORIA: 4096 tokens para soportar archivos Markdown largos
        //let ctx_params = LlamaContextParams::default()
        //    .with_n_ctx(NonZeroU32::new(4096));

        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get() as u32)
            .unwrap_or(4); // Detecta tus núcleos automáticamente

        let threads_to_use = if num_threads >= 16 {
        6  // Máximo absoluto en sistemas de 16+ hilos
        } else if num_threads >= 8 {
            (num_threads as f32 * 0.5).round() as u32  // 50% en sistemas medianos
        } else if num_threads > 2 {
            num_threads - 2  // Comportamiento anterior para sistemas pequeños
        } else {
            num_threads
        };

        // Separamos batch threads (prefill) de inference threads (decode)
        // El batch puede usar un hilo más ya que es una fase puntual, no continua
        let batch_threads = (threads_to_use + 1).min(num_threads);

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(4096))
            .with_n_threads(threads_to_use as i32)
            .with_n_threads_batch(batch_threads as i32);

        let mut ctx = self.model.new_context(&self.backend, ctx_params)
            .map_err(|e| format!("Error creando contexto: {}", e))?;

        let tokens_list = self.model.str_to_token(prompt, AddBos::Always)
            .map_err(|e| format!("Error tokenizando: {}", e))?;

        let mut batch = LlamaBatch::new(4096, 1);
        let last_index = tokens_list.len() - 1;
        
        for (i, &tok) in tokens_list.iter().enumerate() {
            let is_last = i == last_index;
            batch.add(tok, i as i32, &[0], is_last)
                .map_err(|e| format!("Error agregando token al batch: {}", e))?;
        }

        ctx.decode(&mut batch).map_err(|e| format!("Error en decode inicial: {}", e))?;

        let mut sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);

        let mut output = String::new();
        let mut n_cur = batch.n_tokens();

        // Aumentamos a 512 para permitir explicaciones pedagógicas completas
        for _ in 0..512 {
            let current_token = sampler.sample(&ctx, batch.n_tokens() - 1);

            if current_token == self.model.token_eos() {
                break;
            }

            #[allow(deprecated)]
            let token_str = self.model.token_to_str(current_token, Special::Plaintext)
                .unwrap_or_else(|_| "".to_string());
            
            output.push_str(&token_str);

            if token_str.contains("}") {
                break;
            }

            batch.clear();
            batch.add(current_token, n_cur, &[0], true)
                .map_err(|e| format!("Error agregando token generado: {}", e))?;
            
            ctx.decode(&mut batch).map_err(|e| format!("Error decodificando token nuevo: {}", e))?;
            n_cur += 1;
        }

        Ok(output)
    }
    
    fn parse_json_response(&self, raw_text: &str) -> Result<AIResponse, String> {
        let cleaned_text = raw_text.replace("```json", "").replace("```", "");
        let json_start = cleaned_text.find('{').ok_or("No se encontró '{' en la respuesta")?;
        let json_end = cleaned_text.rfind('}').ok_or("No se encontró '}' en la respuesta")? + 1;
        
        let json_str = &cleaned_text[json_start..json_end];
        
        serde_json::from_str(json_str)
            .map_err(|e| format!("Error deserializando AIResponse: {}\nJSON crudo: {}", e, json_str))
    }
}