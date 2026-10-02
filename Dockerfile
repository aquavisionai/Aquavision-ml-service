FROM python:3.14-slim AS runtime
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    AQUAVISION_DEVICE=cpu TORCH_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
COPY requirements.txt requirements-cpu.txt ./
RUN pip install --no-cache-dir -r requirements-cpu.txt

# Keep training tensors out of the final image's layers.
FROM runtime AS weights
COPY best.pt export_weights.py ./
RUN python export_weights.py best.pt inference.pt

FROM runtime AS service
COPY --from=weights /app/inference.pt ./inference.pt
COPY main.py inference.py hd_pipeline.py ./
ENV MODEL_PATH=/app/inference.pt PORT=10000
USER 10001:10001
EXPOSE 10000
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
