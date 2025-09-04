# pull in both Python and Java on top of Debian 11 (Bullseye)
FROM openjdk:17-slim-bullseye

# install Python3 + pip + build tools in one shot
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3.9 python3-pip python3-dev \
      build-essential ca-certificates wget curl unzip git time \
 && update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1 \
 && update-alternatives --install /usr/bin/pip     pip     /usr/bin/pip3     1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
#for faster rebuilds
#COPY . /app

# install everything else via pip
RUN pip install --upgrade pip awscli \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir --no-deps cechmate>=0.0.10 \
 && pip install --no-cache-dir pybind11 setuptools

# Jython + Javaplex
RUN mkdir -p /opt/jython \
 && wget https://repo1.maven.org/maven2/org/python/jython-installer/2.7.3/jython-installer-2.7.3.jar \
 && java -jar jython-installer-2.7.3.jar -s -d /opt/jython \
 && rm jython-installer-2.7.3.jar

ENV PATH="/opt/jython/bin:${PATH}"
ENV CLASSPATH="/app/High_order_TS_with_scaffold/javaplex/javaplex.jar"
ENV ENDPOINT_URL="https://s3.braingeneers.gi.ucsc.edu"

# phat, etc.
RUN wget https://files.pythonhosted.org/packages/43/82/c14de81dc2953a71a060f72f2bc34c41996307956b162751f2a47e2c78f7/phat-1.5.0a.tar.gz#sha256=51e7fe5e05adf5c7e0895765572fff05b979731234251f13011610d71d4980ab \
 && tar -xzf phat-1.5.0a.tar.gz \
 && cd phat-* && python setup.py install \
 && cd /app && rm -rf phat-* phat-1.5.0a.tar.gz

# sanity check
RUN /opt/jython/bin/jython -c "import edu.stanford.math.plex4; print('Javaplex import success')"

#for faster rebuilds
COPY . /app

ENTRYPOINT ["/bin/bash", "-lc"]
CMD ["python", "High_order_TS_with_scaffold/simplicial_multivariate.py", "--help"]
