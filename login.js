import PocketBase from 'https://cdn.jsdelivr.net/npm/pocketbase@0.27.3/dist/pocketbase.es.mjs';

const pb = new PocketBase('https://pb.auravo.ai');
let otpId = null;   // set once requestOTP() succeeds, used by the verify step

// ─── Redirect if already logged in ───────────────────────────────────────────
if (pb.authStore.isValid) {
  window.location.replace('/app');
}

// ─── Check whether OTP is enabled server-side ─────────────────────────────────
// Sign In (password) and Google were deliberately removed from this page — OTP
// is the only sign-*in* method now; Sign Up (password) remains as the only way
// to create a brand new account, since PocketBase's OTP auth requires an
// existing record and doesn't create one on its own.
(async () => {
  try {
    const methods = await pb.collection('users').listAuthMethods();
    if (!methods.otp?.enabled) {
      const tab = document.getElementById('tabOtp');
      tab.disabled = true;
      tab.title = 'Email code sign-in is not configured on the server.';
      showBanner('Email code sign-in is not available right now.', 'error');
    }
  } catch (err) {
    console.error('[Auth] Cannot reach PocketBase:', err);
    showBanner('Cannot reach the auth server. Check your connection.', 'error');
  }
})();

// ─── Tab switching ────────────────────────────────────────────────────────────
window.switchTab = function (tab) {
  document.getElementById('tabOtp').classList.toggle('active',    tab === 'otp');
  document.getElementById('tabSignUp').classList.toggle('active', tab === 'signup');
  document.getElementById('formOtp').classList.toggle('hidden',    tab !== 'otp');
  document.getElementById('formSignUp').classList.toggle('hidden', tab !== 'signup');
  if (tab === 'otp') resetOtpForm();
  hideBanner();
};

// ─── Banner helpers ───────────────────────────────────────────────────────────
function showBanner(msg, type = 'error') {
  const el = document.getElementById('authBanner');
  el.textContent = msg;
  el.className   = `auth-banner ${type}`;
}
function hideBanner() {
  document.getElementById('authBanner').className = 'auth-banner hidden';
}

// ─── Button loading state ─────────────────────────────────────────────────────
function setLoading(btnId, loading) {
  const btn = document.getElementById(btnId);
  btn.disabled = loading;
  btn.querySelector('.btn-text').classList.toggle('hidden',  loading);
  btn.querySelector('.btn-spinner').classList.toggle('hidden', !loading);
}

// ─── Sign Up ──────────────────────────────────────────────────────────────────
// The only remaining password-based flow — creates a new account, then signs the
// user in immediately via password once (OTP requires an existing record, so it
// can't be used for the very first login right after account creation).
window.handleSignUp = async function (e) {
  e.preventDefault();
  hideBanner();

  const name     = document.getElementById('suName').value.trim();
  const email    = document.getElementById('suEmail').value.trim();
  const password = document.getElementById('suPassword').value;
  const confirm  = document.getElementById('suConfirm').value;

  if (password !== confirm) { showBanner('Passwords do not match.'); return; }

  setLoading('btnSignUp', true);

  try {
    await pb.collection('users').create({ name, email, password, passwordConfirm: confirm });
    await pb.collection('users').authWithPassword(email, password);
    window.location.replace('/app');
  } catch (err) {
    showBanner(parseError(err));
    setLoading('btnSignUp', false);
  }
};

// ─── Email Code (OTP) ─────────────────────────────────────────────────────────
// Two-step form reusing the same submit handler: step 1 requests a code (email
// only, hits requestOTP), step 2 verifies it (code only, hits authWithOTP). Which
// step we're in is read off whether the code field is currently hidden.
window.handleOtpSubmit = async function (e) {
  e.preventDefault();
  hideBanner();

  const codeField = document.getElementById('otpCodeField');
  const requestingCode = codeField.classList.contains('hidden');

  if (requestingCode) {
    const email = document.getElementById('otpEmail').value.trim();
    setLoading('btnOtp', true);
    try {
      const result = await pb.collection('users').requestOTP(email);
      otpId = result.otpId;
      document.getElementById('otpEmailField').classList.add('hidden');
      codeField.classList.remove('hidden');
      document.getElementById('otpHint').textContent =
        `Code sent to ${email} — check your inbox (it expires in a few minutes).`;
      document.querySelector('#btnOtp .btn-text').textContent = 'Verify & Sign In';
      document.getElementById('btnOtpReset').classList.remove('hidden');
      document.getElementById('otpCode').focus();
    } catch (err) {
      showBanner(parseError(err));
    } finally {
      setLoading('btnOtp', false);
    }
    return;
  }

  const code = document.getElementById('otpCode').value.trim();
  if (!code) { showBanner('Please enter the code from your email.'); return; }

  setLoading('btnOtp', true);
  try {
    await pb.collection('users').authWithOTP(otpId, code);
    window.location.replace('/app');
  } catch (err) {
    showBanner(parseError(err, 'otp'));
    setLoading('btnOtp', false);
  }
};

window.resetOtpForm = function () {
  otpId = null;
  document.getElementById('otpEmailField').classList.remove('hidden');
  document.getElementById('otpCodeField').classList.add('hidden');
  document.getElementById('otpCode').value = '';
  document.getElementById('otpHint').textContent = "We'll email you a one-time code — no password needed.";
  document.querySelector('#btnOtp .btn-text').textContent = 'Send Code';
  document.getElementById('btnOtpReset').classList.add('hidden');
  hideBanner();
};

// ─── Error parser ─────────────────────────────────────────────────────────────
function parseError(err, context = 'password') {
  if (err?.response?.data) {
    const fields = Object.values(err.response.data);
    if (fields.length) return fields[0]?.message || err.message;
  }
  if (err?.message?.includes('Failed to authenticate')) {
    return context === 'otp' ? 'Invalid or expired code. Please try again.' : 'Invalid email or password.';
  }
  return err?.message || 'Something went wrong. Please try again.';
}
