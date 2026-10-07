//! MacauSplat quality/track-aware training preparation (QVC v1).
//! Originals, calibrated cameras and final PLYs are never overwritten here.
use std::{collections::{HashMap,HashSet},fs,io::Write,path::{Component,Path}};
use image::imageops::FilterType;
use serde::{Serialize,Deserialize};
use sha2::{Digest,Sha256};
use crate::error::{Result,SplatError};
use super::colmap::{SparseReconstruction,Image};
pub const POLICY_VERSION:u32=1;

#[derive(Clone,Debug,Serialize,Deserialize)]
#[serde(rename_all="camelCase")]
pub struct FrameAssessment {pub name:String,pub focus:f64,pub clipped_fraction:f64,pub support:usize,pub selected:bool,pub reason:String}
#[derive(Clone,Debug,Serialize,Deserialize)]
#[serde(rename_all="camelCase")]
pub struct PrecisionReport {
 pub policy_version:u32,pub source_fingerprint:String,pub input_views:usize,pub selected_views:usize,
 pub input_points:usize,pub retained_points:usize,pub reprojection_cutoff_pixels:f64,
 pub median_reprojection_before:Option<f64>,pub median_reprojection_after:Option<f64>,
 pub rejected_point_filter:bool,pub frames:Vec<FrameAssessment>,pub scope:String,
}
#[derive(Clone)]struct ImageRecord{id:u32,name:String,bytes:Vec<u8>,observations:usize,count:usize}
#[derive(Clone)]struct PointRecord{id:i64,error:f64,bytes:Vec<u8>,track:Vec<(u32,u32)>}
fn invalid(message:&str)->SplatError{SplatError::Process(format!("精度增强：{message}"))}
fn u64_at(b:&[u8],i:usize)->Result<u64>{let s=b.get(i..i+8).ok_or_else(||invalid("COLMAP 数据被截断"))?;Ok(u64::from_le_bytes(s.try_into().unwrap()))}
fn u32_at(b:&[u8],i:usize)->Result<u32>{let s=b.get(i..i+4).ok_or_else(||invalid("COLMAP 数据被截断"))?;Ok(u32::from_le_bytes(s.try_into().unwrap()))}
fn safe_name(name:&str)->bool{!name.is_empty()&&Path::new(name).components().all(|c|matches!(c,Component::Normal(_)))&&!name.contains('\\')&&!name.contains(':')}
fn images(b:&[u8])->Result<Vec<ImageRecord>>{
 let n=u64_at(b,0)? as usize;if n>b.len()/73{return Err(invalid("COLMAP 图像数量无效"));}
 let mut p=8;let mut result=Vec::with_capacity(n);
 for _ in 0..n{
  let start=p;let id=u32_at(b,p)?;p+=64;
  let end=b.get(p..).ok_or_else(||invalid("图像记录不完整"))?.iter().position(|v|*v==0).filter(|n|*n<32768).ok_or_else(||invalid("图像名称无效"))?+p;
  let name=String::from_utf8(b[p..end].to_vec()).map_err(|_|invalid("图像名称不是 UTF-8"))?;if !safe_name(&name){return Err(invalid("图像路径不安全"));}
  p=end+1;let count=u64_at(b,p)? as usize;p+=8;let observations=p;
  p=p.checked_add(count.checked_mul(24).ok_or_else(||invalid("观测数量无效"))?).filter(|p|*p<=b.len()).ok_or_else(||invalid("观测数据不完整"))?;
  result.push(ImageRecord{id,name,bytes:b[start..p].to_vec(),observations:observations-start,count});
 }
 if p!=b.len(){return Err(invalid("COLMAP 图像记录长度不一致"));}Ok(result)
}
fn points(b:&[u8])->Result<Vec<PointRecord>>{
 let n=u64_at(b,0)? as usize;if n>b.len()/51{return Err(invalid("COLMAP 点数量无效"));}let mut p=8;let mut result=Vec::with_capacity(n);
 for _ in 0..n{
  let start=p;let id=u64_at(b,p)? as i64;let error=f64::from_bits(u64_at(b,p+35)?);let count=u64_at(b,p+43)? as usize;p+=51;
  let end=p.checked_add(count.checked_mul(8).ok_or_else(||invalid("轨迹数量无效"))?).filter(|p|*p<=b.len()).ok_or_else(||invalid("轨迹数据不完整"))?;
  let mut track=Vec::with_capacity(count);while p<end{track.push((u32_at(b,p)?,u32_at(b,p+4)?));p+=8;}
  result.push(PointRecord{id,error,bytes:b[start..start+43].to_vec(),track});
 }if p!=b.len(){return Err(invalid("COLMAP 点记录长度不一致"));}Ok(result)
}
fn percentile(v:&[f64],q:f64)->Option<f64>{if v.is_empty(){return None;}let mut s=v.to_vec();s.sort_by(f64::total_cmp);Some(s[((s.len()-1)as f64*q).round()as usize])}
pub fn focus_score(gray:&image::GrayImage)->(f64,f64){
 let(w,h)=gray.dimensions();if w<3||h<3{return(0.,1.);}let mut sum=0.;let mut square=0.;let mut lap=0.;let mut clipped=0.;let n=(w*h)as f64;
 for y in 0..h{for x in 0..w{let v=gray.get_pixel(x,y)[0]as f64/255.;sum+=v;square+=v*v;if !(0.02..0.98).contains(&v){clipped+=1.;}if x>0&&x<w-1&&y>0&&y<h-1{let l=(gray.get_pixel(x-1,y)[0]as f64+gray.get_pixel(x+1,y)[0]as f64+gray.get_pixel(x,y-1)[0]as f64+gray.get_pixel(x,y+1)[0]as f64-4.*gray.get_pixel(x,y)[0]as f64)/255.;lap+=l*l;}}}
 (lap/((w-2)*(h-2))as f64/((square/n-(sum/n).powi(2)).max(0.)+0.0025),clipped/n)
}
fn nearby(a:&Image,b:&Image,scene_scale:f64)->bool{
 let ac=a.camera_center();let bc=b.camera_center();let d=ac.iter().zip(bc).map(|(a,b)|(a-b).powi(2)).sum::<f64>().sqrt();
 let ar=a.rotation()[2];let br=b.rotation()[2];d<=scene_scale*0.04&&ar.iter().zip(br).map(|(a,b)|a*b).sum::<f64>()>0.985
}
fn removable(id:u32,support:&HashMap<u32,HashSet<i64>>,counts:&HashMap<i64,usize>)->bool{
 let Some(visible)=support.get(&id)else{return false;};!visible.is_empty()&&visible.iter().all(|p|counts.get(p).copied().unwrap_or(0)>=3)
}
pub fn prepare(dataset:&Path,output:&Path)->Result<PrecisionReport>{prepare_cancellable(dataset,output,||false)}
pub fn prepare_cancellable(dataset:&Path,output:&Path,cancelled:impl Fn()->bool)->Result<PrecisionReport>{
 if output.exists(){return Err(invalid("目标训练集已经存在；请使用新目录"));}
 let model=dataset.join("sparse/0");let camera_bytes=fs::read(model.join("cameras.bin"))?;let image_bytes=fs::read(model.join("images.bin"))?;let point_bytes=fs::read(model.join("points3D.bin"))?;
 let mut digest=Sha256::new();digest.update(&camera_bytes);digest.update(&image_bytes);digest.update(&point_bytes);
 let records=images(&image_bytes)?;let point_records=points(&point_bytes)?;let reconstruction=SparseReconstruction::load_dir(&model).map_err(|e|invalid(&e.to_string()))?;
 let poses:HashMap<_,_>=reconstruction.images.iter().map(|v|(v.id,v)).collect();
 let registered:HashSet<_>=records.iter().map(|r|r.id).collect();if registered.len()!=records.len(){return Err(invalid("重复图像 ID"));}
 let mut frames=Vec::new();let mut support:HashMap<u32,HashSet<i64>>=HashMap::new();let mut counts=HashMap::new();
 for p in &point_records{let ids:HashSet<_>=p.track.iter().map(|t|t.0).filter(|id|registered.contains(id)).collect();counts.insert(p.id,ids.len());for id in ids{support.entry(id).or_default().insert(p.id);}}
 for r in &records{
  if cancelled(){return Err(SplatError::Cancelled);}
  let file=dataset.join("images").join(&r.name);let bytes=fs::read(&file)?;digest.update(r.name.as_bytes());digest.update(&bytes);
  let img=image::load_from_memory(&bytes).map_err(|e|invalid(&format!("无法读取 {}：{e}",r.name)))?;
  let gray=img.resize(384,384,FilterType::Triangle).to_luma8();let(focus,clipped_fraction)=focus_score(&gray);
  frames.push(FrameAssessment{name:r.name.clone(),focus,clipped_fraction,support:support.get(&r.id).map_or(0,HashSet::len),selected:true,reason:"保留视角覆盖".into()});
 }
 let finite_errors:Vec<_>=point_records.iter().map(|p|p.error).filter(|v|v.is_finite()&&*v>=0.).collect();let median=percentile(&finite_errors,0.5).unwrap_or(1.);
 let deviations:Vec<_>=finite_errors.iter().map(|v|(v-median).abs()).collect();let cutoff=(median+4.5*1.4826*percentile(&deviations,0.5).unwrap_or(0.)).clamp(2.,8.);
 // Pair distances avoid assuming either a metric scale or world origin.
 let scale=if reconstruction.images.len()>1{let first=reconstruction.images[0].camera_center();percentile(&reconstruction.images.iter().map(|p|p.camera_center().iter().zip(first).map(|(a,b)|(a-b).powi(2)).sum::<f64>().sqrt()).collect::<Vec<_>>(),0.75).unwrap_or(1.).max(1e-6)}else{1.};
 let typical=percentile(&frames.iter().map(|f|f.focus).collect::<Vec<_>>(),0.5).unwrap_or(0.);
 let mut order:Vec<_>=(0..frames.len()).collect();order.sort_by(|a,b|frames[*a].focus.total_cmp(&frames[*b].focus));let mut kept=records.len();
 for i in order{
  if cancelled(){return Err(SplatError::Cancelled);}
  if kept<=30||kept*5<=records.len()*4||frames[i].focus>=typical*0.22{continue;}
  let Some(pose)=poses.get(&records[i].id)else{continue;};
  let alternative=records.iter().enumerate().any(|(j,r)|i!=j&&frames[j].selected&&frames[j].focus>frames[i].focus*2.&&frames[j].clipped_fraction<0.65&&poses.get(&r.id).is_some_and(|other|nearby(pose,other,scale)));
  if alternative&&removable(records[i].id,&support,&counts){frames[i].selected=false;frames[i].reason="明显模糊，邻近清晰视角可替代且每个观测点仍有至少两视角".into();kept-=1;for id in &support[&records[i].id]{*counts.get_mut(id).unwrap()-=1;}}
 }
 let selected:HashSet<_>=records.iter().zip(&frames).filter(|(_,f)|f.selected).map(|(r,_)|r.id).collect();
 let proposed:HashSet<_>=point_records.iter().filter(|p|p.error.is_finite()&&p.error>=0.&&p.error<=cutoff&&(0..3).all(|k|u64_at(&p.bytes,8+k*8).is_ok_and(|v|f64::from_bits(v).is_finite()))&&p.track.iter().map(|t|t.0).filter(|id|selected.contains(id)).collect::<HashSet<_>>().len()>=3).map(|p|p.id).collect();
 // If the scene is underconstrained, keep every original seed instead of destructive filtering.
 let rejected_filter=proposed.len()<100||proposed.len()*5<point_records.len()*4;
 let point_ids:HashSet<_>=if rejected_filter{point_records.iter().map(|p|p.id).collect()}else{proposed};
 fs::create_dir_all(output.join("images"))?;fs::create_dir_all(output.join("sparse/0"))?;let sparse=output.join("sparse/0");fs::write(sparse.join("cameras.bin"),camera_bytes)?;
 let mut writer=fs::File::create(sparse.join("images.bin"))?;writer.write_all(&(selected.len()as u64).to_le_bytes())?;
 for r in &records{if cancelled(){return Err(SplatError::Cancelled);}if !selected.contains(&r.id){continue;}let mut bytes=r.bytes.clone();for k in 0..r.count{let off=r.observations+k*24+16;let id=u64_at(&bytes,off)?as i64;if id!=-1&&!point_ids.contains(&id){bytes[off..off+8].copy_from_slice(&(-1_i64).to_le_bytes());}}writer.write_all(&bytes)?;
  let source=dataset.join("images").join(&r.name);let target=output.join("images").join(&r.name);fs::create_dir_all(target.parent().unwrap())?;if fs::hard_link(&source,&target).is_err(){fs::copy(source,target)?;}
 }
 let mut writer=fs::File::create(sparse.join("points3D.bin"))?;writer.write_all(&(point_ids.len()as u64).to_le_bytes())?;let mut errors_after=Vec::new();
 for p in &point_records{if !point_ids.contains(&p.id){continue;}let track:Vec<_>=p.track.iter().filter(|t|selected.contains(&t.0)).collect();writer.write_all(&p.bytes)?;writer.write_all(&(track.len()as u64).to_le_bytes())?;for(id,idx)in track{writer.write_all(&id.to_le_bytes())?;writer.write_all(&idx.to_le_bytes())?;}if p.error.is_finite(){errors_after.push(p.error);}}
 let report=PrecisionReport{policy_version:POLICY_VERSION,source_fingerprint:format!("{:x}",digest.finalize()),input_views:records.len(),selected_views:selected.len(),input_points:point_records.len(),retained_points:point_ids.len(),reprojection_cutoff_pixels:cutoff,median_reprojection_before:percentile(&finite_errors,0.5),median_reprojection_after:percentile(&errors_after,0.5),rejected_point_filter:rejected_filter,frames,scope:"训练准备的质量与轨迹检查；重投影误差是像素残差，不能当作厘米精度或最终新视角质量。原始素材与重建保留。".into()};
 fs::write(output.join("precision-report.json"),serde_json::to_vec_pretty(&report)?)?;Ok(report)
}

#[cfg(test)]mod tests{
 use super::*;
 #[test]fn focus_detects_blur_without_rewarding_uniform_white(){let sharp=image::GrayImage::from_fn(96,96,|x,y|image::Luma([if(x/4+y/4)%2==0{30}else{220}]));let blurred=image::imageops::blur(&sharp,3.);assert!(focus_score(&sharp).0>focus_score(&blurred).0*5.);assert_eq!(focus_score(&image::GrayImage::from_pixel(96,96,image::Luma([255]))),(0.,1.));}
 #[test]fn sparse_support_cannot_remove_a_unique_view(){let s=[(1,[10,11].into_iter().collect())].into_iter().collect();assert!(!removable(1,&s,&[(10,3),(11,2)].into_iter().collect()));assert!(removable(1,&s,&[(10,3),(11,4)].into_iter().collect()));}
 #[test]fn record_readers_reject_truncation_and_traversal(){assert!(points(&100u64.to_le_bytes()).is_err());let mut b=Vec::new();b.extend(1u64.to_le_bytes());b.extend([0u8;64]);b.extend(b"../other.png\0");b.extend(0u64.to_le_bytes());assert!(images(&b).is_err());}
 #[test]fn range_statistics_and_names_are_deterministic(){assert_eq!(percentile(&[3.,1.,2.],0.5),Some(2.));assert!(!safe_name("C:/image.png"));assert!(!safe_name("a/../b.png"));assert!(safe_name("folder/frame.png"));}
}

#[cfg(test)]mod format_tests{
 use super::*;
 #[test]fn writes_a_consistent_separate_colmap_dataset(){
  let temp=tempfile::tempdir().unwrap();let input=temp.path().join("input");let model=input.join("sparse/0");fs::create_dir_all(&model).unwrap();fs::create_dir_all(input.join("images")).unwrap();
  let mut cameras=Vec::new();cameras.extend(1u64.to_le_bytes());cameras.extend(1u32.to_le_bytes());cameras.extend(1i32.to_le_bytes());cameras.extend(32u64.to_le_bytes());cameras.extend(32u64.to_le_bytes());for value in [30f64,30.,16.,16.]{cameras.extend(value.to_le_bytes());}fs::write(model.join("cameras.bin"),cameras).unwrap();
  let mut ib=Vec::new();ib.extend(3u64.to_le_bytes());
  for id in 1u32..=3{ib.extend(id.to_le_bytes());for v in [1f64,0.,0.,0.,id as f64*0.1,0.,0.]{ib.extend(v.to_le_bytes());}ib.extend(1u32.to_le_bytes());let name=format!("{id}.png");ib.extend(name.as_bytes());ib.push(0);ib.extend(1u64.to_le_bytes());ib.extend(16f64.to_le_bytes());ib.extend(16f64.to_le_bytes());ib.extend(1i64.to_le_bytes());image::RgbImage::from_pixel(32,32,image::Rgb([120,130,140])).save(input.join("images").join(name)).unwrap();}
  fs::write(model.join("images.bin"),&ib).unwrap();
  let mut pb=Vec::new();pb.extend(1u64.to_le_bytes());pb.extend(1u64.to_le_bytes());for v in [0f64,0.,3.]{pb.extend(v.to_le_bytes());}pb.extend([100,110,120]);pb.extend(0.5f64.to_le_bytes());pb.extend(3u64.to_le_bytes());for id in 1u32..=3{pb.extend(id.to_le_bytes());pb.extend(0u32.to_le_bytes());}fs::write(model.join("points3D.bin"),&pb).unwrap();
  let output=temp.path().join("optimized");let report=prepare(&input,&output).unwrap();assert_eq!(report.selected_views,3);assert_eq!(report.retained_points,1);assert!(report.rejected_point_filter);let scene=SparseReconstruction::load(&output).unwrap();assert_eq!(scene.images.len(),3);assert_eq!(scene.points[0].track.len(),3);assert_eq!(fs::read(model.join("images.bin")).unwrap(),ib);assert_eq!(fs::read(model.join("points3D.bin")).unwrap(),pb);assert!(prepare(&input,&output).is_err());
 }
}
