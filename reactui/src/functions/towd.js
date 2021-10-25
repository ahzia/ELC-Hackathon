import ssw from './ssw';

const swuChange = (oned) => {
    var signtext = oned.trim();
    var signs = ssw.paragraph(signtext);
    return signs;
}

export default swuChange;