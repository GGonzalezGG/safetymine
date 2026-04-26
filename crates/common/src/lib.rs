use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
// En crates/common/src/lib.rs

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "PascalCase")] 
pub enum EppItem {
    Casco,
    Lentes,
    Guantes,
    Respirador,
    ChalecoReflectante,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DetectionResult {
    pub items_detectados: Vec<EppItem>,
    pub timestamp: u64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MinaContext {
    pub nombre_mina: String,
    pub riesgos_activos: Vec<String>, // Ej: ["Gases", "Derrumbes"]
    pub epp_obligatorio: Vec<EppItem>,
    pub path_protocolo: String,      // Ruta al archivo .md
}