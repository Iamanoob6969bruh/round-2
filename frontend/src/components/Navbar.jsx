import { NavLink } from 'react-router-dom';
import { motion } from 'framer-motion';
import TextType from './TextType';

const NAV = [
  { to: '/', label: 'Analyze' },
  { to: '/analytics', label: 'Analytics' },
  { to: '/evaluation', label: 'Evaluation' },
  { to: '/about', label: 'About' },
];

export default function Navbar() {
  return (
    <nav className="fixed top-8 left-1/2 -translate-x-1/2 z-50 w-[99%] max-w-[1600px] flex items-center justify-between px-[15%] py-24 bg-slate-950/15 backdrop-blur-2xl rounded-full shadow-[0_4px_30px_rgba(0,0,0,0.4)] select-none">
      {/* Left: Logo */}
      <div className="flex items-center gap-4">
        <svg className="w-12 h-12" version="1.1" id="Layer_1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" 
             viewBox="0 0 512 512" xml:space="preserve">
          <polygon fill="#FF9911" points="11.084,370.598 11.084,353.719 256.001,212.317 500.916,353.719 500.916,370.598 256.001,512 "/>
          <path id="SVGCleanerId_0" fill="#F18700" d="M382.247,336.943l-77.291-310.99h-0.009 c-0.941-6.455-5.655-12.757-14.162-17.669c-19.131-11.045-50.435-11.045-69.566,0c-8.507,4.912-13.222,11.214-14.162,17.669h-0.009 l-77.292,310.99c-9.104,24.084,2.475,50.31,34.74,68.938c50.328,29.057,132.683,29.057,183.012,0 C379.77,387.253,391.35,361.026,382.247,336.943z"/>
          <g>
            <path id="SVGCleanerId_0_1_" fill="#F18700" d="M382.247,336.943l-77.291-310.99h-0.009 c-0.941-6.455-5.655-12.757-14.162-17.669c-19.131-11.045-50.435-11.045-69.566,0c-8.507,4.912-13.222,11.214-14.162,17.669h-0.009 l-77.292,310.99c-9.104,24.084,2.475,50.31,34.74,68.938c50.328,29.057,132.683,29.057,183.012,0 C379.77,387.253,391.35,361.026,382.247,336.943z"/>
          </g>
          <polygon fill="#F18700" points="256.001,495.121 256.001,512 11.084,370.598 11.084,353.719 "/>
          <polygon fill="#D07400" points="256.001,495.121 256.001,512 500.916,370.598 500.916,353.719 "/>
          <path fill="#FF9911" d="M221.218,48.448c-19.131-11.045-19.131-29.119,0-40.164s50.435-11.045,69.566,0 c19.131,11.045,19.131,29.119,0,40.164C271.653,59.493,240.348,59.493,221.218,48.448z"/>
          <path fill="#D07400" d="M235.13,40.415c-11.478-6.627-11.478-17.471,0-24.098s30.261-6.627,41.74,0 c11.478,6.627,11.478,17.471,0,24.098S246.609,47.042,235.13,40.415z"/>
          <path fill="#E0E0E2" d="M362.967,259.374l-26.3-105.819c-3.277,2.736-7.004,5.342-11.215,7.773 c-38.2,22.054-100.708,22.054-138.907,0c-4.211-2.431-7.937-5.037-11.215-7.774l0,0l-26.3,105.819 c5.071,5.56,11.202,10.804,18.412,15.594c48.705,32.36,128.406,32.36,177.111,0C351.765,270.178,357.898,264.933,362.967,259.374z"/>
        </svg>
        <TextType 
          text={["Traffic VioLens", "AI Vision System"]}
          typingSpeed={140}
          pauseDuration={2500}
          showCursor
          cursorCharacter="█"
          deletingSpeed={30}
          loop={true}
          cursorBlinkDuration={0.6}
          className="text-2xl font-bold tracking-wider text-white uppercase font-sans"
        />
      </div>


      {/* Right: Links */}
      <div className="flex items-center gap-14">
        {NAV.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === '/'}
            className={({ isActive }) => 
              `relative px-6 py-4 text-xl font-semibold tracking-wide transition-all duration-300 ${
                isActive 
                  ? 'text-white' 
                  : 'text-slate-400 hover:text-white'
              }`
            }
          >
            {({ isActive }) => (
              <>
                {isActive && (
                  <motion.div
                    layoutId="active-underline"
                    className="absolute bottom-0 left-0 right-0 h-[4px] bg-gradient-to-r from-indigo-500 to-violet-500 rounded-full"
                    transition={{ type: 'spring', stiffness: 380, damping: 30 }}
                  />
                )}
                {item.label}
              </>
            )}
          </NavLink>
        ))}
      </div>
    </nav>
  );
}
