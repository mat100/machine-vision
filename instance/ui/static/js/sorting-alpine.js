function sortingData() {
    return {
        stats: {},
        images: [],
        classes: [],
        loading: false,
        filters: {
            status: '',
            class: ''
        },
        selectedImages: [],
        bulkClass: '',

        async init() {
            // Load classes from the page
            this.classes = window.classes || [];
            await this.loadData();
        },

        async loadData() {
            this.loading = true;
            try {
                // Load stats and images in parallel
                const [statsResponse, imagesResponse] = await Promise.all([
                    fetch('/api/dataset-stats'),
                    fetch('/api/recent-images?limit=100')
                ]);

                if (statsResponse.ok) {
                    this.stats = await statsResponse.json();
                }

                if (imagesResponse.ok) {
                    const data = await imagesResponse.json();
                    this.images = data.images || [];
                }
            } catch (error) {
                console.error('Error loading data:', error);
                this.showToast('Error loading data', 'error');
            } finally {
                this.loading = false;
            }
        },

        applyFilters() {
            // This would filter the images based on status and class
            // For now, just reload the data
            this.loadData();
        },

        selectImage(img, imagePath) {
            const checkbox = img.parentElement.querySelector('input[type="checkbox"]');
            if (checkbox) {
                checkbox.checked = !checkbox.checked;
                this.updateSelection();
            }
        },

        updateSelection() {
            const checkboxes = document.querySelectorAll('#images-container input[type="checkbox"]:checked');
            this.selectedImages = Array.from(checkboxes).map(cb => cb.dataset.imagePath);
        },

        selectAll() {
            const checkboxes = document.querySelectorAll('#images-container input[type="checkbox"]');
            checkboxes.forEach(cb => cb.checked = true);
            this.updateSelection();
        },

        deselectAll() {
            const checkboxes = document.querySelectorAll('#images-container input[type="checkbox"]');
            checkboxes.forEach(cb => cb.checked = false);
            this.updateSelection();
        },

        async classifySelected() {
            if (!this.bulkClass || this.selectedImages.length === 0) return;

            try {
                const promises = this.selectedImages.map(imagePath =>
                    fetch('/api/classify-image', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/x-www-form-urlencoded',
                        },
                        body: `image_path=${encodeURIComponent(imagePath)}&class_name=${encodeURIComponent(this.bulkClass)}`
                    })
                );

                const results = await Promise.all(promises);
                const successCount = results.filter(r => r.ok).length;

                if (successCount > 0) {
                    this.showToast(`Successfully classified ${successCount} images`, 'success');
                    await this.loadData(); // Reload data
                    this.selectedImages = [];
                    this.bulkClass = '';
                } else {
                    this.showToast('Failed to classify images', 'error');
                }
            } catch (error) {
                console.error('Error classifying images:', error);
                this.showToast('Error classifying images', 'error');
            }
        },

        async deleteSelected() {
            if (this.selectedImages.length === 0) return;

            if (!confirm(`Are you sure you want to delete ${this.selectedImages.length} images?`)) {
                return;
            }

            try {
                const promises = this.selectedImages.map(imagePath =>
                    fetch('/api/delete-image', {
                        method: 'DELETE',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ image_path: imagePath })
                    })
                );

                const results = await Promise.all(promises);
                const successCount = results.filter(r => r.ok).length;

                if (successCount > 0) {
                    this.showToast(`Successfully deleted ${successCount} images`, 'success');
                    await this.loadData(); // Reload data
                    this.selectedImages = [];
                } else {
                    this.showToast('Failed to delete images', 'error');
                }
            } catch (error) {
                console.error('Error deleting images:', error);
                this.showToast('Error deleting images', 'error');
            }
        },

        async deleteSingleImage(imagePath, filename) {
            if (!confirm(`Are you sure you want to delete "${filename}"?`)) {
                return;
            }

            try {
                const response = await fetch('/api/delete-image', {
                    method: 'DELETE',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ image_path: imagePath })
                });

                if (response.ok) {
                    this.showToast(`Successfully deleted "${filename}"`, 'success');
                    await this.loadData(); // Reload data
                    
                    // Remove from selected images if it was selected
                    this.selectedImages = this.selectedImages.filter(path => path !== imagePath);
                } else {
                    this.showToast(`Failed to delete "${filename}"`, 'error');
                }
            } catch (error) {
                console.error('Error deleting image:', error);
                this.showToast(`Error deleting "${filename}"`, 'error');
            }
        },

        showToast(message, type) {
            // Simple toast implementation
            const toast = document.getElementById('toast');
            const messageEl = document.getElementById('toast-message');
            const iconEl = document.getElementById('toast-icon');

            if (messageEl) messageEl.textContent = message;
            
            if (iconEl) {
                if (type === 'success') {
                    iconEl.className = 'h-5 w-5 text-green-400';
                    iconEl.innerHTML = '<path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"></path>';
                } else if (type === 'error') {
                    iconEl.className = 'h-5 w-5 text-red-400';
                    iconEl.innerHTML = '<path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"></path>';
                } else {
                    iconEl.className = 'h-5 w-5 text-blue-400';
                    iconEl.innerHTML = '<path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clip-rule="evenodd"></path>';
                }
            }

            if (toast) {
                toast.classList.remove('hidden');
                setTimeout(() => {
                    toast.classList.add('hidden');
                }, 3000);
            }
        }
    }
} 