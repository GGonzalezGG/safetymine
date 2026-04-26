use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "PascalCase")] // Necesario para que el JSON haga match con el Enum
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
    pub path_protocolo: String,       // Ruta al archivo .md
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AIResponse {
    pub is_safe: bool,
    pub severity: String, // Ej: "OK", "ADVERTENCIA", "CRITICO"
    pub explanation: String,
}