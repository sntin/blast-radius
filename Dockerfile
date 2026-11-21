# Blast Radius entry. Same base image the commits were labelled in.
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90
ENV PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY requirements.lock /app/requirements.lock
RUN pip install --no-cache-dir -r /app/requirements.lock
# the base image ships the standard library without bytecode: compile it once here, not in every run
RUN python -m compileall -q -j 4 /usr/local/lib/python3.13
# sqlglot at the pinned commit (its tests and fixtures are run against each patch)
RUN python -c "import urllib.request; urllib.request.urlretrieve('https://github.com/tobymao/sqlglot/archive/eb11aec4be8cb2122afe713a4816662c3f0fb2fa.tar.gz', '/tmp/sqlglot.tar.gz')" \
    && mkdir -p /opt/sqlglot && tar -xzf /tmp/sqlglot.tar.gz -C /opt/sqlglot --strip-components=1 \
    && rm /tmp/sqlglot.tar.gz \
    && python -m compileall -q -j 4 /opt/sqlglot/sqlglot /opt/sqlglot/tests \
    && chmod -R a+rX /opt/sqlglot
COPY br /app/br
COPY data /app/data
RUN python /app/br/main.py --prepare /app/data/static.pickle \
    && python -m compileall -q /app/br && chmod -R a+rX /app
ENV SKIP_INTEGRATION=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1 VECLIB_MAXIMUM_THREADS=1
# run main from its compiled module rather than as a script (a script is compiled on every start)
CMD ["python", "-c", "import sys; sys.path.insert(0, '/app/br'); import main; main.main()"]
