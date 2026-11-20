# 004 · Ramas y comparación de arquitecturas — tareas

Modelo sugerido: Sonnet implementa; Opus hace T4.2 (capas de atención) y T4.12 (auditoría). Duración: 3 días.

- [ ] T4.1 `modelos.py`: `codificador_por_transito`, `m4_escalares`, `m1_astronet`. Pruebas de formas.
- [ ] T4.2 `MediaEnmascarada`, `AtencionAgregadora`, `m2` con los tres agregadores. Pruebas de invariancia y de máscara.
- [ ] T4.3 `aumentacion.py` con pruebas (la máscara se actualiza; mínimo 4 tránsitos; reproducible con la semilla).
- [ ] T4.4 `exp_ramas.py --solo cordura`: E0 para M1, M2a y M4. **Si falla, detenerse.**
- [ ] T4.5 E1: M1 y M4 (3 semillas). Dejar corriendo.
- [ ] T4.6 [P] `linea_base_demo.py` y `data/dataset_demo.npz`; E1 para M0 (curva y completo).
- [ ] T4.7 E2: M2p, M2g, M2a (3 semillas). Dejar corriendo de noche.
- [ ] T4.8 Selección del agregador (RF-4) → `seleccion_rama2.json`.
- [ ] T4.9 Exportación v1 de `vistas`, `transitos`, `escalares`. `verificar_contrato.py`. Etiqueta `ramas-v1`.
- [ ] T4.10 `exp_sesgos.py`: E3. Comprobar RF-2.
- [ ] T4.11 Tabla `p1_arquitecturas.csv` y pesos de atención de 12 ejemplos al azar.
- [ ] T4.12 Auditoría (Opus): revisar historiales (¿alguna corrida no convergió o paró en la época 1?), buscar AUC > 0,99, confirmar que la selección no usó la prueba.
- [ ] T4.13 (B) E4. 
- [ ] T4.14 (B) E5: variantes de pooling y robustez al desplazamiento.
- [ ] T4.15 (B) E2b y E1b.
- [ ] T4.16 Commit `PT4: ramas`, bitácora con la tabla de resultados tal como salió.

**Si hay retraso, cortar en este orden:** T4.15, T4.14, T4.13, tercera semilla.
