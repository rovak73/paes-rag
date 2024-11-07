# PAES RAG Question Answering System

## Descripción del Proyecto
Sistema de Respuestas Automatizadas para Preguntas de la PAES (Prueba de Acceso a la Educación Superior) utilizando Recuperación Aumentada por Generación (RAG).

## Características
- Extracción de texto de PDFs de exámenes PAES
- Indexación de preguntas y respuestas usando ChromaDB
- Generación de respuestas detalladas con OpenAI
- Interfaz de línea de comandos para consultas

## Requisitos
- Python 3.8+
- OpenAI API Key
- Dependencias en `requirements.txt`

## Configuración Inicial

### 1. Clonar el Repositorio
```bash
git clone https://github.com/tu_usuario/paes-rag.git
cd paes-rag
```

### 2. Configurar Entorno Virtual
```bash
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 3. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 4. Configurar API Key de OpenAI
- Editar el archivo `.env`
- Reemplazar `your_openai_api_key_here` con tu API Key de OpenAI

## Uso

### Ejecutar la Aplicación
```bash
python -m src.main
```

### Ejecutar Pruebas
```bash
pytest tests/
```

## Estructura del Proyecto
```
paes-rag/
│
├── src/
│   ├── main.py           # Módulo principal de la aplicación
│   ├── pdf_processor.py  # Procesamiento de PDFs
│   └── config.py         # Configuraciones y variables de entorno
│
├── tests/
│   ├── test_main.py
│   └── test_pdf_processor.py
│
├── data/
│   └── 2024-23-11-28-paes-regular-oficial-competencia-lectora-p2024.pdf
│
├── requirements.txt
├── .env
└── README.md
```

## Contribuciones
- Reportar issues en GitHub
- Pull requests son bienvenidos

## Licencia
[Especificar la licencia, por ejemplo MIT]

## Contacto
[Información de contacto o correo electrónico]
