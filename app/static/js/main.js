// Global Variables
let currentUser = null;
let token = null;

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
    // Load user data from localStorage
    loadUserData();
    
    // Initialize theme
    initTheme();
    
    // Initialize mobile menu
    initMobileMenu();
    
    // Initialize search
    initSearch();
    
    // Initialize notifications
    initNotifications();
});

// Load user data from localStorage
function loadUserData() {
    const userData = localStorage.getItem('user');
    const storedToken = localStorage.getItem('token');
    
    if (userData && storedToken) {
        currentUser = JSON.parse(userData);
        token = storedToken;
        
        // Update UI with user data
        updateUserUI();
        
        // Setup WebSocket connection
        setupWebSocket();
    } else {
        // Redirect to login if not authenticated and not on auth pages
        const authPages = ['/login', '/register', '/forgot-password'];
        const currentPage = window.location.pathname;
        
        if (!authPages.includes(currentPage)) {
            window.location.href = '/login';
        }
    }
}

// Initialize theme
function initTheme() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    
    if (savedTheme === 'dark' || (savedTheme === 'auto' && prefersDark)) {
        enableDarkMode();
    } else {
        enableLightMode();
    }
    
    // Add theme toggle if exists
    const themeToggle = document.getElementById('themeToggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', toggleTheme);
    }
}

// Toggle theme
function toggleTheme() {
    if (document.body.classList.contains('dark-mode')) {
        enableLightMode();
        localStorage.setItem('theme', 'light');
    } else {
        enableDarkMode();
        localStorage.setItem('theme', 'dark');
    }
}

// Enable dark mode
function enableDarkMode() {
    document.body.classList.add('dark-mode');
    document.documentElement.style.colorScheme = 'dark';
    
    const themeToggle = document.getElementById('themeToggle');
    if (themeToggle) {
        themeToggle.innerHTML = '<i class="fas fa-sun"></i>';
    }
}

// Enable light mode
function enableLightMode() {
    document.body.classList.remove('dark-mode');
    document.documentElement.style.colorScheme = 'light';
    
    const themeToggle = document.getElementById('themeToggle');
    if (themeToggle) {
        themeToggle.innerHTML = '<i class="fas fa-moon"></i>';
    }
}

// Initialize mobile menu
function initMobileMenu() {
    const mobileMenuToggle = document.getElementById('mobileMenuToggle');
    const mobileMenuClose = document.getElementById('mobileMenuClose');
    const mobileMenu = document.getElementById('mobileMenu');
    
    if (mobileMenuToggle && mobileMenu) {
        mobileMenuToggle.addEventListener('click', function() {
            mobileMenu.classList.add('show');
            document.body.style.overflow = 'hidden';
        });
    }
    
    if (mobileMenuClose && mobileMenu) {
        mobileMenuClose.addEventListener('click', function() {
            mobileMenu.classList.remove('show');
            document.body.style.overflow = '';
        });
    }
    
    // Close menu when clicking outside
    document.addEventListener('click', function(event) {
        if (mobileMenu && mobileMenu.classList.contains('show')) {
            if (!mobileMenu.contains(event.target) && 
                !mobileMenuToggle.contains(event.target)) {
                mobileMenu.classList.remove('show');
                document.body.style.overflow = '';
            }
        }
    });
}

// Initialize search
function initSearch() {
    const searchInput = document.getElementById('searchInput');
    if (searchInput) {
        let searchTimeout;
        
        searchInput.addEventListener('input', function() {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(performSearch, 300);
        });
        
        searchInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                performSearch();
            }
        });
    }
}

// Perform search
async function performSearch() {
    const searchInput = document.getElementById('searchInput');
    const query = searchInput.value.trim();
    
    if (query.length < 2) return;
    
    try {
        const response = await fetch(`/api/search?q=${encodeURIComponent(query)}`, {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (response.ok) {
            const results = await response.json();
            showSearchResults(results);
        }
    } catch (error) {
        console.error('Search error:', error);
    }
}

// Show search results
function showSearchResults(results) {
    // Create or update search results dropdown
    let dropdown = document.getElementById('searchResultsDropdown');
    
    if (!dropdown) {
        dropdown = document.createElement('div');
        dropdown.id = 'searchResultsDropdown';
        dropdown.className = 'search-results-dropdown';
        document.querySelector('.nav-search').appendChild(dropdown);
    }
    
    if (results.length === 0) {
        dropdown.innerHTML = '<div class="search-result-item">No results found</div>';
    } else {
        dropdown.innerHTML = results.map(result => `
            <a href="/profile/${result.id}" class="search-result-item">
                <img src="/static/uploads/profile_pics/${result.profile_picture}" alt="${result.username}">
                <div>
                    <strong>${result.username}</strong>
                    <span>${result.full_name || ''}</span>
                </div>
            </a>
        `).join('');
    }
    
    dropdown.style.display = 'block';
}

// Initialize notifications
function initNotifications() {
    const notificationBell = document.querySelector('.nav-link[href="/notifications"]');
    if (notificationBell) {
        notificationBell.addEventListener('click', function(e) {
            e.preventDefault();
            showNotifications();
        });
    }
}

// Show notifications
async function showNotifications() {
    try {
        const response = await fetch('/api/notifications', {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (response.ok) {
            const notifications = await response.json();
            createNotificationModal(notifications);
        }
    } catch (error) {
        console.error('Error fetching notifications:', error);
    }
}

// Create notification modal
function createNotificationModal(notifications) {
    // Remove existing modal if exists
    const existingModal = document.getElementById('notificationsModal');
    if (existingModal) {
        existingModal.remove();
    }
    
    const modal = document.createElement('div');
    modal.id = 'notificationsModal';
    modal.className = 'modal';
    modal.innerHTML = `
        <div class="modal-content">
            <div class="modal-header">
                <h2>Notifications</h2>
                <button class="modal-close" onclick="closeModal('notificationsModal')">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            <div class="modal-body">
                <div class="notifications-list">
                    ${notifications.map(notification => `
                        <div class="notification-item ${notification.is_read ? '' : 'unread'}" 
                             data-id="${notification.id}">
                            <div class="notification-icon">
                                <i class="fas fa-${getNotificationIcon(notification.type)}"></i>
                            </div>
                            <div class="notification-content">
                                <p>${notification.message}</p>
                                <span class="notification-time">${formatTime(notification.created_at)}</span>
                            </div>
                            ${!notification.is_read ? 
                                '<button class="mark-read-btn" onclick="markNotificationRead(this)">' +
                                '<i class="fas fa-check"></i></button>' : ''}
                        </div>
                    `).join('')}
                </div>
            </div>
            <div class="modal-footer">
                <button class="btn btn-secondary" onclick="markAllNotificationsRead()">
                    Mark All as Read
                </button>
                <button class="btn btn-primary" onclick="closeModal('notificationsModal')">
                    Close
                </button>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
    modal.style.display = 'flex';
    
    // Close modal when clicking outside
    modal.addEventListener('click', function(e) {
        if (e.target === modal) {
            closeModal('notificationsModal');
        }
    });
}

// Get notification icon based on type
function getNotificationIcon(type) {
    const icons = {
        'like': 'heart',
        'comment': 'comment',
        'follow': 'user-plus',
        'message': 'envelope',
        'mention': 'at',
        'reaction': 'smile'
    };
    return icons[type] || 'bell';
}

// Format time
function formatTime(dateString) {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now - date;
    const diffSec = Math.floor(diffMs / 1000);
    const diffMin = Math.floor(diffSec / 60);
    const diffHour = Math.floor(diffMin / 60);
    const diffDay = Math.floor(diffHour / 24);
    
    if (diffSec < 60) return 'Just now';
    if (diffMin < 60) return `${diffMin}m ago`;
    if (diffHour < 24) return `${diffHour}h ago`;
    if (diffDay < 7) return `${diffDay}d ago`;
    return date.toLocaleDateString();
}

// Mark notification as read
async function markNotificationRead(button) {
    const notificationItem = button.closest('.notification-item');
    const notificationId = notificationItem.dataset.id;
    
    try {
        const response = await fetch(`/api/notifications/${notificationId}/read`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (response.ok) {
            notificationItem.classList.remove('unread');
            button.remove();
            updateNotificationBadge();
        }
    } catch (error) {
        console.error('Error marking notification as read:', error);
    }
}

// Mark all notifications as read
async function markAllNotificationsRead() {
    try {
        const response = await fetch('/api/notifications/read-all', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (response.ok) {
            document.querySelectorAll('.notification-item').forEach(item => {
                item.classList.remove('unread');
                const button = item.querySelector('.mark-read-btn');
                if (button) button.remove();
            });
            updateNotificationBadge();
        }
    } catch (error) {
        console.error('Error marking all notifications as read:', error);
    }
}

// Update notification badge
function updateNotificationBadge() {
    const badges = document.querySelectorAll('.notification-badge');
    const unreadCount = document.querySelectorAll('.notification-item.unread').length;
    
    badges.forEach(badge => {
        if (unreadCount > 0) {
            badge.textContent = unreadCount;
            badge.style.display = 'flex';
        } else {
            badge.style.display = 'none';
        }
    });
}

// Setup WebSocket connection
function setupWebSocket() {
    if (!currentUser) return;
    
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/${currentUser.id}`;
    
    const ws = new WebSocket(wsUrl);
    
    ws.onopen = function() {
        console.log('WebSocket connected');
    };
    
    ws.onmessage = function(event) {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data);
    };
    
    ws.onclose = function() {
        console.log('WebSocket disconnected. Reconnecting...');
        setTimeout(setupWebSocket, 3000);
    };
    
    ws.onerror = function(error) {
        console.error('WebSocket error:', error);
    };
}

// Handle WebSocket messages
function handleWebSocketMessage(data) {
    switch (data.type) {
        case 'notification':
            showToast(data.message, 'info');
            updateNotificationBadge();
            break;
        case 'message':
            showMessageNotification(data);
            break;
        case 'post_update':
            updatePost(data.post_id, data.updates);
            break;
        case 'comment':
            addCommentToPost(data.post_id, data.comment);
            break;
        case 'reaction':
            updateReactions(data.post_id, data.reactions);
            break;
    }
}

// Show message notification
function showMessageNotification(data) {
    if (document.hidden) {
        // Show browser notification
        if (Notification.permission === 'granted') {
            new Notification(`New message from ${data.sender}`, {
                body: data.content,
                icon: data.sender_avatar
            });
        }
    } else {
        // Show toast notification
        showToast(`New message from ${data.sender}: ${data.content}`, 'info');
    }
}

// Update user UI
function updateUserUI() {
    // Update profile pictures
    const profilePics = document.querySelectorAll('.nav-profile-pic, .profile-avatar');
    profilePics.forEach(pic => {
        if (currentUser.profile_picture) {
            pic.src = `/static/uploads/profile_pics/${currentUser.profile_picture}`;
        }
    });
    
    // Update usernames
    const usernames = document.querySelectorAll('.username-display');
    usernames.forEach(element => {
        element.textContent = currentUser.username;
    });
    
    // Update full names
    const fullnames = document.querySelectorAll('.fullname-display');
    fullnames.forEach(element => {
        element.textContent = currentUser.full_name || currentUser.username;
    });
}

// Logout function
function logout() {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    window.location.href = '/login';
}

// Show toast notification
function showToast(message, type = 'info') {
    // Remove existing toasts
    const existingToasts = document.querySelectorAll('.toast');
    existingToasts.forEach(toast => {
        if (toast.parentNode) {
            toast.parentNode.removeChild(toast);
        }
    });
    
    // Create new toast
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
        <i class="fas fa-${getToastIcon(type)}"></i>
        <span>${message}</span>
    `;
    
    document.body.appendChild(toast);
    
    // Remove toast after 3 seconds
    setTimeout(() => {
        if (toast.parentNode) {
            toast.parentNode.removeChild(toast);
        }
    }, 3000);
}

// Get toast icon based on type
function getToastIcon(type) {
    const icons = {
        'success': 'check-circle',
        'error': 'exclamation-circle',
        'warning': 'exclamation-triangle',
        'info': 'info-circle'
    };
    return icons[type] || 'info-circle';
}

// Close modal
function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.style.display = 'none';
    }
}

// Format number with commas
function formatNumber(num) {
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    }
    if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
}

// API request helper
async function apiRequest(url, options = {}) {
    const defaultOptions = {
        headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
        }
    };
    
    const mergedOptions = { ...defaultOptions, ...options };
    
    try {
        const response = await fetch(url, mergedOptions);
        
        if (response.status === 401) {
            // Token expired, redirect to login
            logout();
            return null;
        }
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Request failed');
        }
        
        return await response.json();
    } catch (error) {
        console.error('API request error:', error);
        showToast(error.message || 'Network error', 'error');
        throw error;
    }
}