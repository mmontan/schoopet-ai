// Main entry point

// Import styles if you wanted to manage them here, but we linked in HTML for simplicity.
// import './style.css' 

console.log('Schoopet is ready to remember.');

// Add glass effect to header on scroll
const header = document.querySelector('header');
let isScrolled = false;
let ticking = false;

window.addEventListener('scroll', () => {
  if (!ticking) {
    window.requestAnimationFrame(() => {
      const scrolled = window.scrollY > 50;
      if (scrolled !== isScrolled) {
        isScrolled = scrolled;
        if (isScrolled) {
          header.style.background = 'rgba(5, 5, 7, 0.8)';
          header.style.backdropFilter = 'blur(10px)';
          header.style.borderBottom = '1px solid rgba(255, 255, 255, 0.05)';
        } else {
          header.style.background = 'transparent';
          header.style.backdropFilter = 'none';
          header.style.borderBottom = 'none';
        }
      }
      ticking = false;
    });
    ticking = true;
  }
});
