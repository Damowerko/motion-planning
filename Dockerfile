FROM damowerko/torchcps:latest

RUN mkdir -p /home/$USER/motion-planning
WORKDIR /home/$USER/motion-planning

# install requirements
COPY poetry.lock pyproject.toml README.md ./
RUN poetry install --no-root

# install motion-planning package in editable mode
COPY src/ src/
COPY scripts/ scripts/
RUN poetry install --only-root