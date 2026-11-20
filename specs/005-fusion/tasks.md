# 005 · Fusión y ablación — tareas

Modelo sugerido: Sonnet implementa; Opus revisa T5.7. Duración: 2 días. T5.1–T5.2 se pueden adelantar al bloque B.

- [ ] T5.1 `codificador_imagen_v0.py` → `modelos/imagen/v0/`. `verificar_contrato.py`.
- [ ] T5.2 `vectores.py` con pruebas.
- [ ] T5.3 `fusion.py`: `cabeza`, `entrenar_fusion`. Ensayo con v0.
- [ ] T5.4 Confirmar que `modelos/imagen/v0/` cubre todos los ejemplos no prerregistrados. No se espera el codificador de Visión (DD-13).
- [ ] T5.5 E6a (5 semillas) y E6c.
- [ ] T5.6 `diagnostico_apilado`. Si se activa el ajuste cruzado: implementarlo, correrlo y anotar la desviación.
- [ ] T5.7 Revisión (Opus): ¿la fusión supera a la mejor rama sola? ¿Algún AUC > 0,99? ¿El diagnóstico es coherente?
- [ ] T5.8 E7: 8 configuraciones × 5 semillas.
- [ ] T5.9 Tabla `p2_fusion.csv` y estratos (RF-12 a RF-14).
- [ ] T5.10 (C) E7b. 
- [ ] T5.11 (C) E6b.
- [ ] T5.12 Commit `PT5: fusión`, bitácora con la versión de imagen usada.
