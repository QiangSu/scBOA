FROM mambaorg/micromamba:1.5.10

LABEL maintainer="QiangSu"
LABEL description="scBOA: single-cell Bayesian Optimization and Analysis"

WORKDIR /workspace

COPY --chown=mambauser:mambauser environment-docker.yml /tmp/environment-docker.yml

RUN micromamba create -y -n scboa -f /tmp/environment-docker.yml && \
    micromamba clean --all --yes

COPY --chown=mambauser:mambauser scBOA.py README.md pyproject.toml ./
COPY --chown=mambauser:mambauser src ./src

RUN printf '%s\n' \
'#!/bin/sh' \
'set -e' \
'if command -v micromamba >/dev/null 2>&1; then' \
'  MAMBA_BIN="$(command -v micromamba)"' \
'elif [ -x /bin/micromamba ]; then' \
'  MAMBA_BIN="/bin/micromamba"' \
'else' \
'  echo "ERROR: micromamba not found" >&2' \
'  exit 1' \
'fi' \
'exec "$MAMBA_BIN" run -n scboa python /workspace/scBOA.py "$@"' \
> /workspace/scboa-entrypoint.sh && chmod +x /workspace/scboa-entrypoint.sh

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["/workspace/scboa-entrypoint.sh"]
CMD ["--help"]
