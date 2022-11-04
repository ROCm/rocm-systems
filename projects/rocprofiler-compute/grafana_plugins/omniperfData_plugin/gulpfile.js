const { src, dest, series } = require('gulp');
const babel = require('gulp-babel');

function script() {
	return src('src/**/*.js')
		.pipe(
			babel({
				presets: ['es2015'],
				plugins: [
					'transform-es2015-modules-systemjs',
					'transform-es2015-for-of',
				],
			})
		)
		.pipe(dest('dist'));
}

function srcToDist() {
	return src(['src/**/*', '!src/**/*.js']).pipe(dest('dist'));
}

exports.default = series(script, srcToDist);
