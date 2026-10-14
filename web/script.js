const form=document.querySelector("#news-form");
const textarea=document.querySelector("#news-text");
const count=document.querySelector("#char-count");
const sampleButton=document.querySelector("#sample-btn");
const resultEmpty=document.querySelector("#result-empty");
const resultContent=document.querySelector("#result-content");
const menuToggle=document.querySelector(".menu-toggle");
const nav=document.querySelector(".nav-links");
const sampleText="Scientists have announced a new discovery after a study involving thousands of participants. The researchers say the findings require further independent review before any broad conclusions can be drawn.";
textarea.addEventListener("input",()=>{count.textContent=`${textarea.value.length.toLocaleString()} / 10,000 characters`;});
sampleButton.addEventListener("click",()=>{textarea.value=sampleText;textarea.dispatchEvent(new Event("input"));textarea.focus();});
menuToggle.addEventListener("click",()=>{const open=nav.classList.toggle("open");menuToggle.setAttribute("aria-expanded",String(open));});
nav.querySelectorAll("a").forEach(link=>link.addEventListener("click",()=>{nav.classList.remove("open");menuToggle.setAttribute("aria-expanded","false");}));
form.addEventListener("submit",event=>{
  event.preventDefault();
  const text=textarea.value.trim();
  if(!text){textarea.focus();return;}
  resultEmpty.hidden=true;
  resultContent.hidden=false;
  document.querySelector("#result-label").textContent="Demo preview";
  document.querySelector("#confidence-value").textContent="Not available";
  document.querySelector("#confidence-bar").style.width="0%";
  document.querySelector("#result-explanation").textContent="Your article has been submitted to the interface. Connect this page to the project's Python prediction pipeline to display a real classification and confidence score.";
  document.querySelector("#result-badge").classList.add("demo-state");
  document.querySelector("#result-badge").querySelector("span").style.background="#8b8c86";
  document.querySelector("#result-disclaimer");
  resultContent.scrollIntoView({behavior:"smooth",block:"nearest"});
});