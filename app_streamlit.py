"""Dashboard de visualización — SOLO LECTURA.

NO ejecuta ni mide la pipeline: lee results/*.csv (generados por bench/, medidos con
CUDA Events) y las imágenes de out/. Streamlit jamás toca los números evaluados.

Uso:  streamlit run app_streamlit.py
"""
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "results.csv"
SWEEP = ROOT / "results" / "config_sweep.csv"
ENV = ROOT / "results" / "environment.txt"
CMDS = ROOT / "results" / "commands.log"
OUT = ROOT / "out"

st.set_page_config(page_title="Actividad 4 — Pipeline GPU", layout="wide")
st.title("Pipeline de imágenes en GPU — resultados")
st.caption("Visualización de solo lectura: los tiempos provienen de bench/ "
           "(CUDA Events / perf_counter). Streamlit no mide nada.")


@st.cache_data
def load_csv(path: str, mtime: float) -> pd.DataFrame:
    # mtime en la firma => el cache se invalida cuando el CSV cambia
    return pd.read_csv(path)


def maybe(path: Path) -> pd.DataFrame | None:
    return load_csv(str(path), path.stat().st_mtime) if path.exists() else None


df = maybe(RESULTS)
tab_res, tab_vis, tab_sweep, tab_env = st.tabs(
    ["📊 Resultados", "🖼️ Comparación visual", "🧩 Efecto bloque/tile", "🔬 Entorno y trazabilidad"])

# ------------------------------------------------------------------ Resultados
with tab_res:
    if df is None:
        st.warning("No hay results/results.csv. Corre: `python bench/run_bench.py --versions cpu ...`")
    else:
        c1, c2 = st.columns(2)
        imgs = c1.multiselect("Imágenes", sorted(df["image"].unique()),
                              default=sorted(df["image"].unique()))
        gausses = c2.multiselect("Config gaussian", sorted(df["gauss"].unique()),
                                 default=sorted(df["gauss"].unique()))
        v = df[df["image"].isin(imgs) & df["gauss"].isin(gausses)].copy()
        v["cfg"] = v["image"] + " · " + v["gauss"] + " · x" + v["scale"].astype(str)

        st.subheader("Speed-up vs CPU (tiempo total, incluye transferencias)")
        gpu = v[v["version"] != "cpu"]
        if not gpu.empty:
            ch = (alt.Chart(gpu).mark_bar().encode(
                x=alt.X("cfg:N", title=None, sort=None),
                y=alt.Y("speedup_vs_cpu:Q", title="speed-up (×)"),
                color="version:N", xOffset="version:N",
                tooltip=["version", "cfg", "speedup_vs_cpu", "total_ms_mean"]).properties(height=300))
            st.altair_chart(ch, width='stretch')

        st.subheader("Desglose kernel vs transferencias (dónde se va el tiempo)")
        parts = gpu.melt(id_vars=["version", "cfg"],
                         value_vars=["kernel_ms", "h2d_ms", "d2h_ms"],
                         var_name="componente", value_name="ms").dropna()
        if not parts.empty:
            ch = (alt.Chart(parts).mark_bar().encode(
                x=alt.X("version:N", title=None), y=alt.Y("ms:Q", title="ms"),
                color=alt.Color("componente:N",
                                scale=alt.Scale(domain=["kernel_ms", "h2d_ms", "d2h_ms"],
                                                range=["#4c78a8", "#f58518", "#e45756"])),
                column=alt.Column("cfg:N", title=None, header=alt.Header(labelAngle=-45, labelAlign="right")),
                tooltip=["version", "componente", "ms"]).properties(height=260))
            st.altair_chart(ch)

        st.subheader("Tiempo por etapa")
        stg = v.melt(id_vars=["version", "cfg"],
                     value_vars=["gaussian_ms", "sobel_ms", "resize_ms"],
                     var_name="etapa", value_name="ms").dropna()
        log_scale = st.checkbox("Escala logarítmica", value=True)
        if not stg.empty:
            ch = (alt.Chart(stg).mark_bar().encode(
                x=alt.X("cfg:N", title=None), xOffset="version:N",
                y=alt.Y("ms:Q", title="ms", scale=alt.Scale(type="log") if log_scale else alt.Scale()),
                color="version:N", column=alt.Column("etapa:N", title=None),
                tooltip=["version", "etapa", "ms"]).properties(height=240))
            st.altair_chart(ch)

        st.subheader("Throughput (MPx/s) vs tamaño de imagen")
        thr = v.groupby(["version", "megapixels"], as_index=False)["throughput_mpx_s"].mean()
        ch = (alt.Chart(thr).mark_line(point=True).encode(
            x=alt.X("megapixels:Q", title="megapíxeles"),
            y=alt.Y("throughput_mpx_s:Q", title="MPx/s", scale=alt.Scale(type="log")),
            color="version:N", tooltip=["version", "megapixels", "throughput_mpx_s"]
        ).properties(height=300))
        st.altair_chart(ch, width='stretch')

        st.subheader("Tabla completa")
        st.dataframe(v.drop(columns=["cfg"]), width='stretch', hide_index=True)

# --------------------------------------------------------- Comparación visual
with tab_vis:
    st.caption("Salidas ya generadas por las versiones (out/). El MAE se calcula aquí "
               "solo para visualizar; la validación oficial es bench/validate.py.")
    pngs = sorted(OUT.glob("*.png")) if OUT.exists() else []
    if not pngs:
        st.info("No hay imágenes en out/ aún.")
    else:
        # parsear nombres: <img>__<version>__k#_s#__x#__<etapa>.png
        recs = []
        for p in pngs:
            parts = p.stem.split("__")
            if len(parts) == 5:
                recs.append({"image": parts[0], "version": parts[1], "gauss": parts[2],
                             "scale": parts[3], "stage": parts[4], "path": p})
        cat = pd.DataFrame(recs)
        c1, c2, c3, c4 = st.columns(4)
        si = c1.selectbox("Imagen", sorted(cat["image"].unique()))
        sg = c2.selectbox("Config", sorted(cat[cat["image"] == si]["gauss"].unique()))
        sc = c3.selectbox("Escala", sorted(cat[(cat["image"] == si) & (cat["gauss"] == sg)]["scale"].unique()))
        se = c4.selectbox("Etapa", sorted(cat[(cat["image"] == si) & (cat["gauss"] == sg)]["stage"].unique()))
        sel = cat[(cat["image"] == si) & (cat["gauss"] == sg) & (cat["scale"] == sc) & (cat["stage"] == se)]
        cols = st.columns(max(len(sel), 1))
        ref = sel[sel["version"] == "cpu"]
        ref_arr = np.asarray(Image.open(ref.iloc[0]["path"])).astype(np.float64) if len(ref) else None
        for col, (_, row) in zip(cols, sel.iterrows()):
            with col:
                st.image(str(row["path"]), caption=row["version"], width='stretch')
                if ref_arr is not None and row["version"] != "cpu":
                    arr = np.asarray(Image.open(row["path"])).astype(np.float64)
                    if arr.shape == ref_arr.shape:
                        st.metric("MAE vs CPU", f"{np.abs(arr - ref_arr).mean():.4f}")

# ------------------------------------------------------------- Config sweep
with tab_sweep:
    sw = maybe(SWEEP)
    if sw is None:
        st.info("No hay results/config_sweep.csv. Corre: `python bench/run_config_sweep.py`")
    else:
        st.caption("Efecto del tamaño de bloque (clásico) y del tile (Tile C++/Python) — §8 de la pauta.")
        ch = (alt.Chart(sw).mark_bar().encode(
            x=alt.X("tile:N", title="configuración", sort=None),
            y=alt.Y("kernel_ms:Q", title="tiempo de kernels (ms)"),
            color="version:N", column=alt.Column("version:N", title=None),
            tooltip=["version", "tile", "kernel_ms", "gaussian_ms", "total_ms_mean"]
        ).properties(height=280).resolve_scale(x="independent"))
        st.altair_chart(ch)
        st.dataframe(sw[["version", "tile", "gaussian_ms", "sobel_ms", "resize_ms",
                         "kernel_ms", "h2d_ms", "d2h_ms", "total_ms_mean", "notes"]],
                     width='stretch', hide_index=True)

# ---------------------------------------------------------------- Entorno
with tab_env:
    obs = ROOT / "results" / "profiles" / "OBSERVACIONES.md"
    if obs.exists():
        st.subheader("Profiling (Nsight Systems)")
        st.markdown(obs.read_text(encoding="utf-8"))
        st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Entorno experimental")
        if ENV.exists():
            st.code(ENV.read_text(encoding="utf-8"), language="text")
        else:
            st.info("Corre: `python bench/capture_env.py`")
    with c2:
        st.subheader("Registro de comandos (últimos 40)")
        if CMDS.exists():
            lines = CMDS.read_text(encoding="utf-8").splitlines()
            st.code("\n".join(lines[-40:]), language="text")
        else:
            st.info("Aún no hay commands.log")
