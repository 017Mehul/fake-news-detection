const form=document.querySelector("#news-form");
const textarea=document.querySelector("#news-text");
const count=document.querySelector("#char-count");
const resultEmpty=document.querySelector("#result-empty");
const resultContent=document.querySelector("#result-content");
const menuToggle=document.querySelector(".menu-toggle");
const nav=document.querySelector(".nav-links");
const API_BASE=(window.FAKE_NEWS_API_URL||"http://127.0.0.1:8000").replace(/\/$/,"");
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
function showResult(label, explanation, isError=false, confidence=null){
 resultEmpty.hidden=true;
 resultContent.hidden=false;
 document.querySelector("#result-label").textContent=label;
 const explanationNode=document.querySelector("#result-explanation");
 explanationNode.textContent=explanation;
 resultContent.querySelector(".result-badge").classList.toggle("error",isError);
 resultContent.querySelector(".result-disclaimer").textContent="Machine-learning prediction only — not independent fact verification.";
 let confidenceNode=document.querySelector("#result-confidence");
 if(!confidenceNode){
  confidenceNode=document.createElement("p");
  confidenceNode.id="result-confidence";
  confidenceNode.className="result-confidence";
  explanationNode.insertAdjacentElement("afterend",confidenceNode);
 }
 confidenceNode.textContent=confidence===null?"":"Model confidence: "+confidence+"%";
 resultContent.scrollIntoView({behavior:"smooth",block:"nearest"});
}
form.addEventListener("submit",async event=>{
 event.preventDefault();
 const text=textarea.value.trim();
 if(!text){textarea.focus();return;}
 if(text.length>5000){showResult("Article too long","Please keep the article under 5,000 characters.",true);return;}
 const submitButton=form.querySelector('button[type="submit"],button:not([type])');
 const originalText=submitButton?.textContent;
 if(submitButton){submitButton.disabled=true;submitButton.textContent="Analyzing…";}
 resultEmpty.hidden=true;
 resultContent.hidden=false;
 document.querySelector("#result-label").textContent="Connecting to model…";
 document.querySelector("#result-explanation").textContent="Sending the article to the Python prediction API.";
 try{
  const response=await fetch(API_BASE+"/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({text,model:"logistic_regression"})});
  const payload=await response.json().catch(()=>({}));
  if(!response.ok)throw new Error(typeof payload.detail==="string"?payload.detail:"Prediction API returned HTTP "+response.status+".");
  const prediction=String(payload.prediction||"").toUpperCase();
  if(!["REAL","FAKE"].includes(prediction))throw new Error("The API response did not contain a valid prediction.");
  const confidence=Number.isFinite(payload.confidence_percent)?payload.confidence_percent:null;
  showResult(prediction==="FAKE"?"Likely Fake News":"Likely Real News",
   "The trained "+(payload.model==="naive_bayes"?"Multinomial Naive Bayes":"Logistic Regression")+" model classified this article as "+prediction+". Treat this as a model signal, not proof of truth.",false,confidence);
 }catch(error){
  const message=error instanceof TypeError
   ?"Could not reach the Python API at "+API_BASE+". Start the backend, or set window.FAKE_NEWS_API_URL in web/config.js to your deployed API URL."
   :error.message;
  showResult("Prediction unavailable",message,true);
 }finally{
  if(submitButton){submitButton.disabled=false;submitButton.textContent=originalText||"Check Article";}
 }
});
