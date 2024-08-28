const CACHE_NAME = 'face-score-cache-v1';
const urlsToCache = [
  '/',
  '/styles.css',
  './static/logo_16x16.jpg',
  './static/logo_32x32.jpg',
  './static/logo_256x256.jpg',
  './static/logo_512x512.jpg',
  // Add other resources to cache here
];

// Install event
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => {
        return cache.addAll(urlsToCache);
      })
  );
});

// Fetch event
self.addEventListener('fetch', (event) => {
  event.respondWith(
    caches.match(event.request)
      .then((response) => {
        return response || fetch(event.request);
      })
  );
});
