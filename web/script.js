const form=document.querySelector("#news-form");
const textarea=document.querySelector("#news-text");
const count=document.querySelector("#char-count");
const resultEmpty=document.querySelector("#result-empty");
const resultContent=document.querySelector("#result-content");
const menuToggle=document.querySelector(".menu-toggle");
const nav=document.querySelector(".nav-links");
const MODEL_URL=window.FAKE_NEWS_MODEL_URL||"./model/model.json";
let browserModelPromise=null;
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
function loadBrowserModel(){
 if(!browserModelPromise){
  browserModelPromise=fetch(MODEL_URL).then(response=>{
   if(!response.ok)throw new Error("Browser model file was not found. Generate web/model/model.json with python export_browser_model.py and commit it.");
   return response.json();
  }).then(model=>{
   if(model.format_version!==1||model.model_type!=="tfidf_logistic_regression")throw new Error("Unsupported browser model format.");
   return model;
  }).catch(error=>{browserModelPromise=null;throw error;});
 }
 return browserModelPromise;
}
function predictInBrowser(text,model){
 const counts=new Map();
 const tokenPattern=/\b\w\w+\b/g;
 const stopWords=new Set(model.stop_words||[]);
 const normalized=model.lowercase===false?text:text.toLowerCase();
 const tokens=normalized.match(tokenPattern)||[];
 for(const token of tokens){
  if(stopWords.has(token))continue;
  counts.set(token,(counts.get(token)||0)+1);
 }
 const features=new Map();
 for(const [token,count] of counts){
  const index=model.vocabulary[token];
  if(index===undefined)continue;
  const tf=model.sublinear_tf?1+Math.log(count):count;
  features.set(index,tf*model.idf[index]);
 }
 if(model.norm==="l2"){
  let squared=0;
  for(const value of features.values())squared+=value*value;
  const norm=Math.sqrt(squared);
  if(norm>0)for(const [index,value] of features)features.set(index,value/norm);
 }
 // sklearn's default binary LogisticRegression predicts class 1 when decision > 0.
 let score=model.intercept;
 for(const [index,value] of features)score+=value*model.coefficient[index];
 const fakeProbability=score>=0?1/(1+Math.exp(-score)):Math.exp(score)/(1+Math.exp(score));
 const prediction=score>0?"FAKE":"REAL";
 return {prediction,confidence:Math.max(fakeProbability,1-fakeProbability),fakeProbability,realProbability:1-fakeProbability};
}
function showResult(label, explanation, isError=false, confidence=null){
 resultEmpty.hidden=true;
 resultContent.hidden=false;
 document.querySelector("#result-label").textContent=label;
 const explanationNode=document.querySelector("#result-explanation");
 explanationNode.textContent=explanation;
 resultContent.querySelector(".result-badge").classList.toggle("error",isError);
 resultContent.querySelector(".result-disclaimer").textContent="Browser-side machine-learning classification only — not independent fact verification.";
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
 document.querySelector("#result-label").textContent="Loading model…";
 document.querySelector("#result-explanation").textContent="Loading the trained model into your browser. Article text stays on this device.";
 try{
  const model=await loadBrowserModel();
  const result=predictInBrowser(text,model);
  showResult(result.prediction==="FAKE"?"Likely Fake News":"Likely Real News",
   "The trained Logistic Regression model classified this article as "+result.prediction+". Treat this as a model signal, not proof of truth.",
   false,(result.confidence*100).toFixed(2));
 }catch(error){
  showResult("Model unavailable",error.message+" You can generate the browser model locally; see API_SETUP.md.",true);
 }finally{
  if(submitButton){submitButton.disabled=false;submitButton.textContent=originalText||"Check Article";}
 }
});
