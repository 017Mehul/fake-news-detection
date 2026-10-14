const form=document.querySelector("#news-form");
const textarea=document.querySelector("#news-text");
const count=document.querySelector("#char-count");
const resultEmpty=document.querySelector("#result-empty");
const resultContent=document.querySelector("#result-content");
const menuToggle=document.querySelector(".menu-toggle");
const nav=document.querySelector(".nav-links");
const examples={
 politics:"The election commission announced the final results after completing its verification process. Officials said the published figures are available for public review, and independent observers have been invited to examine the records.",
 technology:"Researchers have introduced a new battery prototype in a laboratory demonstration. The team says further testing, independent replication, and large-scale manufacturing studies are needed before commercial use can be assessed.",
 health:"A new health study reports an association between a lifestyle factor and a measured outcome. The authors caution that the results do not establish causation and recommend additional research before changing public health guidance.",
 entertainment:"The production studio has announced its upcoming release schedule. The dates and details were shared in an official statement, with further information expected through the studio's verified channels."
};
textarea.addEventListener("input",()=>{count.textContent=textarea.value.length+"/5000";});
document.querySelectorAll("[data-sample]").forEach(button=>button.addEventListener("click",()=>{
 textarea.value=examples[button.dataset.sample]||"";
 textarea.dispatchEvent(new Event("input"));
 textarea.focus();
}));
menuToggle.addEventListener("click",()=>{
 const open=nav.classList.toggle("open");
 menuToggle.setAttribute("aria-expanded",String(open));
});
nav.querySelectorAll("a").forEach(link=>link.addEventListener("click",()=>{
 nav.classList.remove("open");
 menuToggle.setAttribute("aria-expanded","false");
}));
form.addEventListener("submit",event=>{
 event.preventDefault();
 if(!textarea.value.trim()){textarea.focus();return;}
 resultEmpty.hidden=true;
 resultContent.hidden=false;
 document.querySelector("#result-label").textContent="Demo preview";
 document.querySelector("#result-explanation").textContent="The article text was entered successfully. The Python prediction backend is not connected yet, so no real classification or confidence score has been generated.";
 resultContent.scrollIntoView({behavior:"smooth",block:"nearest"});
});