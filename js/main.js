// ============================================================
// main.js — Egocentric Homes page switching + interactions
// ============================================================


// ---- Page switching -----------------------------------------

function showPage(id) {
  // Hide all pages
  document.querySelectorAll('.page').forEach(p => p.classList.remove('page--active'));

  // Show the target page
  const target = document.getElementById(id);
  if (target) target.classList.add('page--active');

  // Update nav link active state
  document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('nav-link--active'));
  const activeLink = document.querySelector(`.nav-link[data-page="${id}"]`);
  if (activeLink) activeLink.classList.add('nav-link--active');

  // Scroll to top on page switch
  window.scrollTo({ top: 0 });
}

// Wire up navbar links
document.querySelectorAll('.nav-link[data-page]').forEach(link => {
  link.addEventListener('click', e => {
    e.preventDefault();
    showPage(link.getAttribute('data-page'));
  });
});

// Wire up "Explore Dataset" button if present
document.querySelectorAll('a[href="#dataset"]').forEach(btn => {
  btn.addEventListener('click', e => {
    e.preventDefault();
    showPage('dataset');
  });
});


// ---- Copy-to-clipboard for the Python snippet ---------------

function copyCode() {
  const codeEl = document.getElementById('code-snippet');
  const text = codeEl.innerText;

  navigator.clipboard.writeText(text)
    .then(() => {
      const btn = document.querySelector('.code-block__copy');
      btn.textContent = 'Copied!';
      setTimeout(() => { btn.textContent = 'Copy'; }, 2000);
    })
    .catch(() => {
      alert('Could not copy — please select and copy manually.');
    });
}


// ---- Disabled-button feedback for "coming soon" links -------

document.querySelectorAll('.btn--outlined[href="#"]').forEach(btn => {
  btn.addEventListener('click', e => {
    e.preventDefault();
    btn.style.borderColor = '#2563EB';
    btn.style.color       = '#2563EB';
    setTimeout(() => {
      btn.style.borderColor = '';
      btn.style.color       = '';
    }, 600);
  });
});
