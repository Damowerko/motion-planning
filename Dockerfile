FROM damowerko/torchcps:latest

RUN mkdir -p /home/$USER/motion-planning
WORKDIR /home/$USER/motion-planning
# fix git ownership
RUN git config --global --add safe.directory /home/$USER/motion-planning

# install requirements
COPY poetry.lock pyproject.toml README.md ./
RUN poetry install --no-root

# copy the rest of the repository (including git)
COPY . .
RUN poetry install --only-root