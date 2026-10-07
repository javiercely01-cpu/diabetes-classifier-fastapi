#!/usr/bin/env bash
# Pruebas manuales con curl para la API Diabetes Classifier.
# Requiere el servicio corriendo (ver README):  uvicorn app.main:app --port 8000
# Uso: bash tests/test_api_curl.sh [BASE_URL]

BASE_URL="${1:-http://localhost:8000}"
JSON="Content-Type: application/json"

echo "=== 1) GET /health ==="
curl -s -i "$BASE_URL/health"

echo -e "\n\n=== 2) GET /model-info ==="
curl -s -i "$BASE_URL/model-info"

echo -e "\n\n=== 3) POST /predict - paciente con perfil de riesgo alto ==="
curl -s -i -X POST "$BASE_URL/predict" -H "$JSON" \
  -d '{"pregnancies":6,"glucose":148,"blood_pressure":72,"skin_thickness":35,"insulin":0,"bmi":33.6,"diabetes_pedigree_function":0.627,"age":50}'

echo -e "\n\n=== 4) POST /predict - paciente con perfil de riesgo bajo ==="
curl -s -i -X POST "$BASE_URL/predict" -H "$JSON" \
  -d '{"pregnancies":1,"glucose":85,"blood_pressure":66,"skin_thickness":29,"insulin":0,"bmi":26.6,"diabetes_pedigree_function":0.351,"age":31}'

echo -e "\n\n=== 5) POST /predict - valores 0 (dato no medido, se imputa con la mediana) ==="
curl -s -i -X POST "$BASE_URL/predict" -H "$JSON" \
  -d '{"pregnancies":2,"glucose":0,"blood_pressure":0,"skin_thickness":0,"insulin":0,"bmi":0,"diabetes_pedigree_function":0.5,"age":40}'

echo -e "\n\n=== 6) POST /predict - error de validación (edad negativa) ==="
curl -s -i -X POST "$BASE_URL/predict" -H "$JSON" \
  -d '{"pregnancies":2,"glucose":120,"blood_pressure":70,"skin_thickness":20,"insulin":80,"bmi":30,"diabetes_pedigree_function":0.5,"age":-3}'

echo -e "\n\n=== 7) POST /predict - error (falta el campo bmi) ==="
curl -s -i -X POST "$BASE_URL/predict" -H "$JSON" \
  -d '{"pregnancies":2,"glucose":120,"blood_pressure":70,"skin_thickness":20,"insulin":80,"diabetes_pedigree_function":0.5,"age":40}'

echo -e "\n\n=== 8) POST /predict - error (tipo inválido: glucose como texto) ==="
curl -s -i -X POST "$BASE_URL/predict" -H "$JSON" \
  -d '{"pregnancies":2,"glucose":"alta","blood_pressure":70,"skin_thickness":20,"insulin":80,"bmi":30,"diabetes_pedigree_function":0.5,"age":40}'

echo -e "\n\n=== 9) GET /stats (monitoreo básico) ==="
curl -s -i "$BASE_URL/stats"
echo
