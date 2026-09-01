import PocketBase from 'https://cdn.jsdelivr.net/npm/pocketbase@0.27.3/dist/pocketbase.es.mjs';

const pb = new PocketBase('https://pb.auravo.ai');

// ─── Guard: redirect to login if session is missing or expired ────────────────
if (!pb.authStore.isValid) {
  window.location.replace('/login');
}

// ─── Inject user pill into the app header ────────────────────────────────────
const user     = pb.authStore.record;
const initials = (user?.name || user?.email || '?')[0].toUpperCase();

const pill = document.createElement('div');
pill.className = 'user-pill';
pill.innerHTML = `
  <div class="user-avatar-circle">${initials}</div>
  <span class="user-display">${user?.name || user?.email?.split('@')[0] || 'User'}</span>
  <button class="logout-btn" id="btnLogout">Sign out</button>
`;

document.querySelector('.header-inner')?.appendChild(pill);

document.getElementById('btnLogout').addEventListener('click', () => {
  pb.authStore.clear();
  window.location.replace('/login');
});

// ─── Authenticated fetch ───────────────────────────────────────────────────
// Attaches the PocketBase session token so the backend can identify which
// user is making the request (assessments/history are scoped per-user).
window.authFetch = async function authFetch(url, options = {}) {
  const headers = { ...(options.headers || {}), 'x-pb-token': pb.authStore.token || '' };
  let res = await fetch(url, { ...options, headers });

  if (res.status === 401) {
    try {
      await pb.collection('users').authRefresh();
      headers['x-pb-token'] = pb.authStore.token || '';
      res = await fetch(url, { ...options, headers });
    } catch {
      pb.authStore.clear();
      window.location.replace('/login');
      return res;
    }
    if (res.status === 401) {
      pb.authStore.clear();
      window.location.replace('/login');
    }
  }
  return res;
};
