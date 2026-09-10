import Image from "next/image";
import logo from "@/assets/smart-badminton-logo.png";

export function Brand() {
  return (
    <div className="brand" aria-label="Smart Badminton Studio">
      <span className="brand-logo"><Image src={logo} alt="" preload /></span>
      <div><strong>SMART BADMINTON</strong><span>SEE YOUR GAME DIFFERENTLY</span></div>
    </div>
  );
}
