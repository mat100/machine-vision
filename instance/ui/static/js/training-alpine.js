/**
 * Training page Alpine.js functionality
 */

function trainingData() {
    return {
        models: [],
        datasetStats: {},
        trainingProgress: {},
        trainingConfig: {
            architecture: 'resnet18',
            epochs: 50,
            learning_rate: 0.001,
            batch_size: 32
        },
        isTraining: false,
        activeModelName: null,
        loading: false,
        errorMessage: null,
        successMessage: null,

        init() {
            this.loadData();
            this.startProgressPolling();
        },

        async loadData() {
            this.loading = true;
            try {
                // Load models
                const modelsResponse = await fetch('/api/models');
                if (!modelsResponse.ok) {
                    throw new Error('Failed to load models');
                }
                const modelsData = await modelsResponse.json();
                this.models = modelsData.models || [];

                // Load dataset stats
                const statsResponse = await fetch('/api/dataset-stats');
                if (!statsResponse.ok) {
                    throw new Error('Failed to load dataset stats');
                }
                this.datasetStats = await statsResponse.json();

                // Load training status
                const statusResponse = await fetch('/api/training-status');
                if (!statusResponse.ok) {
                    throw new Error('Failed to load training status');
                }
                this.trainingProgress = await statusResponse.json();
                this.isTraining = this.trainingProgress.status === 'training';

                // Find active model
                const activeModel = this.models.find(m => m.is_active);
                this.activeModelName = activeModel ? activeModel.name : null;

                this.errorMessage = null;
            } catch (error) {
                console.error('Error loading data:', error);
                this.errorMessage = 'Failed to load data: ' + error.message;
            } finally {
                this.loading = false;
            }
        },

        async startTraining() {
            // Validate training requirements
            if (!this.validateTrainingRequirements()) {
                return;
            }

            try {
                this.loading = true;
                this.errorMessage = null;
                this.successMessage = null;

                const response = await fetch('/api/train', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(this.trainingConfig)
                });

                if (response.ok) {
                    this.isTraining = true;
                    this.trainingProgress.status = 'training';
                    this.trainingProgress.message = 'Training started...';
                    this.successMessage = 'Training started successfully!';
                } else {
                    const error = await response.json();
                    throw new Error(error.detail || 'Failed to start training');
                }
            } catch (error) {
                console.error('Error starting training:', error);
                this.errorMessage = 'Failed to start training: ' + error.message;
            } finally {
                this.loading = false;
            }
        },

        validateTrainingRequirements() {
            const requirements = [];
            
            if ((this.datasetStats.classified_images || 0) < 10) {
                requirements.push('At least 10 classified images are required');
            }
            
            if (Object.keys(this.datasetStats.class_distribution || {}).length < 2) {
                requirements.push('At least 2 classes are required');
            }

            if (requirements.length > 0) {
                this.errorMessage = 'Training requirements not met:\n' + requirements.join('\n');
                return false;
            }

            return true;
        },

        async activateModel(modelName) {
            try {
                this.loading = true;
                this.errorMessage = null;
                this.successMessage = null;

                const response = await fetch(`/api/models/${modelName}/activate`, {
                    method: 'POST'
                });

                if (response.ok) {
                    this.successMessage = `Model "${modelName}" activated successfully!`;
                    await this.loadData(); // Reload to update active model
                } else {
                    const error = await response.json();
                    throw new Error(error.detail || 'Failed to activate model');
                }
            } catch (error) {
                console.error('Error activating model:', error);
                this.errorMessage = 'Failed to activate model: ' + error.message;
            } finally {
                this.loading = false;
            }
        },

        async deleteModel(modelName) {
            if (!confirm(`Are you sure you want to delete model "${modelName}"? This action cannot be undone.`)) {
                return;
            }

            try {
                this.loading = true;
                this.errorMessage = null;
                this.successMessage = null;

                const response = await fetch(`/api/models/${modelName}`, {
                    method: 'DELETE'
                });

                if (response.ok) {
                    this.successMessage = `Model "${modelName}" deleted successfully!`;
                    await this.loadData(); // Reload to update models list
                } else {
                    const error = await response.json();
                    throw new Error(error.detail || 'Failed to delete model');
                }
            } catch (error) {
                console.error('Error deleting model:', error);
                this.errorMessage = 'Failed to delete model: ' + error.message;
            } finally {
                this.loading = false;
            }
        },

        startProgressPolling() {
            // Poll training progress every 2 seconds when training
            setInterval(() => {
                if (this.isTraining) {
                    this.updateTrainingProgress();
                }
            }, 2000);
        },

        async updateTrainingProgress() {
            try {
                const response = await fetch('/api/training-status');
                if (!response.ok) {
                    throw new Error('Failed to fetch training status');
                }
                
                this.trainingProgress = await response.json();
                
                if (this.trainingProgress.status === 'completed') {
                    this.isTraining = false;
                    this.successMessage = 'Training completed successfully!';
                    await this.loadData(); // Reload models after training completes
                } else if (this.trainingProgress.status === 'failed') {
                    this.isTraining = false;
                    this.errorMessage = 'Training failed: ' + (this.trainingProgress.message || 'Unknown error');
                    await this.loadData(); // Reload models after training fails
                }
            } catch (error) {
                console.error('Error updating training progress:', error);
            }
        },

        clearMessages() {
            this.errorMessage = null;
            this.successMessage = null;
        },

        formatFileSize(bytes) {
            if (!bytes) return 'Unknown size';
            const sizes = ['B', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(1024));
            return (bytes / Math.pow(1024, i)).toFixed(2) + ' ' + sizes[i];
        },

        formatDate(dateString) {
            return new Date(dateString).toLocaleDateString('en-US', {
                year: 'numeric',
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
            });
        },

        getTrainingStatusColor(status) {
            switch (status) {
                case 'training': return 'text-blue-600';
                case 'completed': return 'text-green-600';
                case 'failed': return 'text-red-600';
                default: return 'text-gray-600';
            }
        },

        getTrainingStatusIcon(status) {
            switch (status) {
                case 'training': return 'spinner';
                case 'completed': return 'check-circle';
                case 'failed': return 'x-circle';
                default: return 'clock';
            }
        }
    }
} 