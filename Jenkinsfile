// Runs lint and tests inside the Docker image so validation is reproducible.
// Works on Linux (sh) and Windows (bat) Jenkins agents.
def run(String command) {
    if (isUnix()) { sh command } else { bat command }
}

pipeline {
    agent any

    environment {
        IMAGE = "ai-qa-agent-platform:${env.BUILD_NUMBER}"
    }

    stages {
        stage('Build image') {
            steps {
                run "docker build -t ${IMAGE} ."
            }
        }

        stage('Lint') {
            steps {
                run "docker run --rm ${IMAGE} ruff check app tests"
            }
        }

        stage('Test') {
            steps {
                run "docker run --rm -e LLM_ENABLED=false -v \"${env.WORKSPACE}/reports:/app/reports\" ${IMAGE} pytest -q --junitxml=reports/junit.xml"
            }
        }
    }

    post {
        always {
            junit allowEmptyResults: true, testResults: 'reports/junit.xml'
            archiveArtifacts allowEmptyArchive: true, artifacts: 'reports/**'
        }
    }
}
