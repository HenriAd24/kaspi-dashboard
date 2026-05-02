/* Kaspi Dashboard – Service Worker
 * Strategy:
 *   API calls  → Network-first, cache fallback (stale data shown offline)
 *   HTML/static → Cache-first after first load
 */

const CACHE   = "kaspi-v5";
const API_RE  = /\/api\//;
const PRECACHE = ["/"];

self.addEventListener("install", e => {
  e.waitUntil(
    caches.open(CACHE)
      .then(c => c.addAll(PRECACHE))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", e => {
  e.waitUntil(
    caches.keys()
      .then(keys => Promise.all(
        keys.filter(k => k !== CACHE).map(k => caches.delete(k))
      ))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", e => {
  // Only handle same-origin requests
  if (!e.request.url.startsWith(self.location.origin)) return;

  if (API_RE.test(e.request.url)) {
    // Network-first: fresh data when online, stale when offline
    e.respondWith(
      fetch(e.request.clone())
        .then(resp => {
          const copy = resp.clone();
          caches.open(CACHE).then(c => c.put(e.request, copy));
          return resp;
        })
        .catch(() => caches.match(e.request))
    );
  } else {
    // Cache-first: HTML and static assets
    e.respondWith(
      caches.match(e.request).then(cached => {
        const network = fetch(e.request).then(resp => {
          caches.open(CACHE).then(c => c.put(e.request, resp.clone()));
          return resp;
        });
        return cached || network;
      })
    );
  }
});

// Notify clients when new content is available
self.addEventListener("message", e => {
  if (e.data && e.data.type === "SKIP_WAITING") self.skipWaiting();
});
