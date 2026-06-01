// ============================================================
// main.js — HomeHands dataset page interactive behaviour
// ============================================================


// ---- Copy-to-clipboard for the Python snippet ---------------

// Called by the onclick="copyCode()" attribute on the Copy button
function copyCode() {
  // Grab the text content of the <code> element inside the code block
  const codeEl = document.getElementById('code-snippet');
  const text = codeEl.innerText; // Raw plain text — no HTML tags

  // Clipboard API is async; handle success and failure separately
  navigator.clipboard.writeText(text)
    .then(() => {
      // On success: temporarily change button label to "Copied!"
      const btn = document.querySelector('.code-block__copy');
      btn.textContent = 'Copied!';              // Update label
      setTimeout(() => { btn.textContent = 'Copy'; }, 2000); // Reset after 2s
    })
    .catch(() => {
      // Fallback for browsers that block clipboard access
      alert('Could not copy — please select and copy manually.');
    });
}


// ---- Smooth scrolling for all anchor links ------------------

// Select every <a> whose href starts with "#" (internal page links)
document.querySelectorAll('a[href^="#"]').forEach(link => {
  link.addEventListener('click', e => {
    // Resolve the href to an actual DOM element
    const targetId = link.getAttribute('href'); // e.g. "#dataset"
    const target   = document.querySelector(targetId); // DOM element or null

    if (target) {
      e.preventDefault(); // Stop the browser's default instant-jump behaviour
      // Smooth scroll to the section; 'start' aligns top of section with viewport top
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  });
});


// ---- Active nav-link highlight driven by IntersectionObserver ---

// Collect all sections with an id (the ones nav links point to)
const sections = document.querySelectorAll('section[id]');

// Collect all nav link <a> elements
const navLinks = document.querySelectorAll('.navbar__links a');

// Helper: remove the active accent color from every nav link
function clearActiveLinks() {
  navLinks.forEach(l => {
    l.style.color = '';        // Reset inline style so CSS takes over
    l.style.fontWeight = '';   // Reset any weight override
  });
}

// IntersectionObserver fires when a section enters or leaves the viewport
const sectionObserver = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    // Only act on sections that are currently intersecting (visible)
    if (entry.isIntersecting) {
      clearActiveLinks(); // Remove highlight from all links first

      // Find the nav link whose href matches the visible section's id
      const activeLink = document.querySelector(
        `.navbar__links a[href="#${entry.target.id}"]`
      );

      if (activeLink) {
        activeLink.style.color      = '#FF4500'; // Accent color for active link
        activeLink.style.fontWeight = '700';     // Slightly bolder when active
      }
    }
  });
}, {
  // rootMargin shifts the detection zone: fires when section is near viewport center
  rootMargin: '-30% 0px -60% 0px',
  threshold: 0 // Trigger as soon as any pixel of the section is in the detection zone
});

// Attach the observer to each section
sections.forEach(section => sectionObserver.observe(section));


// ---- Disabled-button feedback for "coming soon" links -------

// Select all outlined buttons that link to "#" (placeholder links)
document.querySelectorAll('.btn--outlined[href="#"]').forEach(btn => {
  btn.addEventListener('click', e => {
    e.preventDefault(); // Prevent page jump to top

    // Flash the button border briefly to communicate "not yet available"
    btn.style.borderColor = '#FF4500'; // Accent border flash
    btn.style.color       = '#FF4500'; // Accent text flash
    setTimeout(() => {
      btn.style.borderColor = ''; // Reset after 600ms
      btn.style.color       = ''; // Reset after 600ms
    }, 600);
  });
});
